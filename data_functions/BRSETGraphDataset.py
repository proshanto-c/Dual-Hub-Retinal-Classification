import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, List, Dict, Union, Optional, Callable

def patchify(image: torch.Tensor, patch_size: int):
    if image.dim() != 3:
        raise ValueError("Expected image tensor shaped [C, H, W].")

    c, h, w = image.shape
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h))
        _, h, w = image.shape

    grid_h = h // patch_size
    grid_w = w // patch_size

    patches = (
        image.unfold(1, patch_size, patch_size)
        .unfold(2, patch_size, patch_size)
        .permute(1, 2, 0, 3, 4)
        .contiguous()
    ).view(grid_h * grid_w, c * patch_size * patch_size)

    coords = []
    for r in range(grid_h):
        for c_ in range(grid_w):
            coords.append(
                [
                    0.0 if grid_h == 1 else r / float(grid_h - 1),
                    0.0 if grid_w == 1 else c_ / float(grid_w - 1),
                ]
            )
    coord_tensor = torch.tensor(coords, dtype=patches.dtype, device=patches.device)
    return patches, coord_tensor, grid_h, grid_w


def grid_edge_index(grid_h: int, grid_w: int, connectivity: Any = "fc", device=None) -> torch.Tensor:
    n_patches = grid_h * grid_w
    conn_str = str(connectivity).lower()
    
    if conn_str in ("fc", "fully_connected", "all", "0"):
        # Fully Connected (FC) patch graph: every patch connects to every other patch
        src = torch.arange(n_patches, device=device).repeat_interleave(n_patches)
        dst = torch.arange(n_patches, device=device).repeat(n_patches)
        mask = src != dst
        return torch.stack([src[mask], dst[mask]], dim=0)
    elif connectivity in (4, 8, "4", "8"):
        c_val = int(connectivity)
        neighbors_4 = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        neighbors_8 = neighbors_4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        neighbors = neighbors_8 if c_val == 8 else neighbors_4

        def idx(r: int, c: int) -> int:
            return r * grid_w + c

        src, dst = [], []
        for r in range(grid_h):
            for c in range(grid_w):
                i = idx(r, c)
                for dr, dc in neighbors:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < grid_h and 0 <= cc < grid_w:
                        src.append(i)
                        dst.append(idx(rr, cc))

        return torch.tensor([src, dst], dtype=torch.long, device=device)
    else:
        # Default fallback to FC
        src = torch.arange(n_patches, device=device).repeat_interleave(n_patches)
        dst = torch.arange(n_patches, device=device).repeat(n_patches)
        mask = src != dst
        return torch.stack([src[mask], dst[mask]], dim=0)

class BRSETGraphDataset(Dataset):
    """
    Heterogeneous Graph Dataset for Patient-Level or Image-Level BRSET predictions.
    """
    def __init__(
        self,
        data: Union[Path, str, pd.DataFrame],
        image_dir: Path,
        target_cols: List[str],
        demo_cols: List[str],
        demo_specs: List[Any], 
        image_size: int = 256,
        patch_size: int = 32,
        connectivity: Any = "fc",
        transform: Optional[Callable] = None,
        validate_paths: bool = True,
        prediction_level: str = "patient",
        train_frame: Optional[pd.DataFrame] = None,
        label_corr_matrix: Optional[torch.Tensor] = None,
        demo_corr_matrix: Optional[torch.Tensor] = None,
    ) -> None:
        
        if prediction_level not in ["patient", "image", "dual_hub", "central_hub"]:
            raise ValueError("prediction_level must be either 'patient', 'image', 'dual_hub', or 'central_hub'")
            
        if isinstance(data, (Path, str)):
            self.frame = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.frame = data.copy()
        else:
            raise ValueError("data must be a path-like string/Path or a pandas DataFrame.")
            
        self.image_dir = Path(image_dir)
        self.target_cols = target_cols
        self.demo_cols = demo_cols
        self.demo_specs = demo_specs
        self.train_frame = train_frame
        self.preset_label_corr = label_corr_matrix
        self.preset_demo_corr = demo_corr_matrix
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.connectivity = connectivity
        self.transform = transform
        self.prediction_level = prediction_level
        
        self._cache_vision_edges() 
        self._cache_correlations()
        self.samples = []
        
        # Updated to handle branching graph logic
        self._build_samples(validate_paths)

    def _cache_vision_edges(self) -> None:
        """Pre-calculates the static grid edges to prevent redundant math in __getitem__."""
        self.grid_h = self.image_size // self.patch_size
        self.grid_w = self.image_size // self.patch_size
        self.n_patches = self.grid_h * self.grid_w
        
        # 1. Cache Spatial Adjacency
        adj = grid_edge_index(self.grid_h, self.grid_w, self.connectivity, device=torch.device('cpu'))
        self.base_patch_adj_src = adj[0]
        self.base_patch_adj_dst = adj[1]
        
        # Calculate Gaussian distance weight for patch adjacency
        src_y, src_x = self.base_patch_adj_src // self.grid_w, self.base_patch_adj_src % self.grid_w
        dst_y, dst_x = self.base_patch_adj_dst // self.grid_w, self.base_patch_adj_dst % self.grid_w
        dist_sq = (src_y - dst_y)**2 + (src_x - dst_x)**2
        sigma_sq = (self.grid_w / 2.0)**2  # Gaussian spread heuristic
        self.base_patch_adj_attr = torch.exp(-dist_sq / sigma_sq).unsqueeze(-1).to(torch.float32)
        
        # 2. Cache Hierarchical Adjacency
        self.base_patch_to_img_src = torch.arange(self.n_patches, dtype=torch.long)
        self.base_patch_to_img_dst = torch.zeros(self.n_patches, dtype=torch.long)

    def _cache_correlations(self) -> None:
        """Calculates Pearson correlation matrices for labels and demographics strictly from the training split."""
        if self.preset_label_corr is not None and self.preset_demo_corr is not None:
            self.label_corr_matrix = self.preset_label_corr
            self.demo_corr_matrix = self.preset_demo_corr
            return

        ref_df = self.train_frame
        if ref_df is None:
            raise ValueError(
                "BRSETGraphDataset strictly requires `train_frame` to compute Pearson correlation matrices "
                "from the training split. Passing non-training data without `train_frame` is strictly forbidden."
            )

        # 1. Label correlation
        if self.preset_label_corr is not None:
            self.label_corr_matrix = self.preset_label_corr
        else:
            if len(self.target_cols) > 1:
                lbl_df = ref_df[self.target_cols].astype(float)
                corr = lbl_df.corr(method="pearson").fillna(0.0).values
            else:
                corr = np.array([[1.0]])
            self.label_corr_matrix = torch.tensor(corr, dtype=torch.float32)
        
        # 2. Demographic correlation
        if self.preset_demo_corr is not None:
            self.demo_corr_matrix = self.preset_demo_corr
        else:
            demo_names = [s.name for s in self.demo_specs]
            if len(demo_names) > 1:
                demo_df = ref_df[demo_names].copy()
                for s in self.demo_specs:
                    if s.kind == "categorical":
                        demo_df[s.name] = demo_df[s.name].astype("category").cat.codes
                    else:
                        if demo_df[s.name].dtype == object:
                            demo_df[s.name] = demo_df[s.name].str.replace(',', '.', regex=False)
                        demo_df[s.name] = pd.to_numeric(demo_df[s.name], errors="coerce")
                demo_corr = demo_df.corr(method="pearson").fillna(0.0).values
            else:
                demo_corr = np.array([[1.0]])
            self.demo_corr_matrix = torch.tensor(demo_corr, dtype=torch.float32)

    def _build_samples(self, validate: bool) -> None:
        """Constructs either multi-image patient hubs or single-image standalone graphs."""
        
        if self.prediction_level == "patient":
            # --- BATCH BY PATIENT ---
            for patient_id, group in self.frame.groupby("patient_id", sort=True):
                image_paths = []
                eye_ids = []
                
                for _, row in group.iterrows():
                    raw_path = row.get("image_path", f"{row['image_id']}.jpg")
                    filename = Path(raw_path).name
                    if self.image_dir and (self.image_dir / filename).is_file():
                        p = self.image_dir / filename
                    else:
                        p = Path(raw_path) if Path(raw_path).is_absolute() else self.image_dir / filename
                    
                    if validate and not p.is_file():
                        raise FileNotFoundError(f"Missing image: {p}")
                    
                    image_paths.append(p)
                    eye_val = row.get("exam_eye", -1)
                    eye_ids.append(int(float(eye_val)) if pd.notna(eye_val) else -1)
                
                label_vector = group[self.target_cols].max().to_numpy(dtype=np.float32)
                demographic_values = {col: group[col].dropna().mode()[0] if not group[col].dropna().empty else None for col in self.demo_cols}
                
                self.samples.append({
                    "patient_id": str(patient_id),
                    "image_paths": image_paths,
                    "eye_ids": eye_ids,
                    "label_vector": label_vector,
                    "demographics": demographic_values
                })
                
        else:
            # --- BATCH BY IMAGE ---
            for _, row in self.frame.iterrows():
                raw_path = row.get("image_path", f"{row['image_id']}.jpg")
                filename = Path(raw_path).name
                if self.image_dir and (self.image_dir / filename).is_file():
                    p = self.image_dir / filename
                else:
                    p = Path(raw_path) if Path(raw_path).is_absolute() else self.image_dir / filename
                
                if validate and not p.is_file():
                    raise FileNotFoundError(f"Missing image: {p}")
                
                eye_val = row.get("exam_eye", -1)
                eye_id = int(float(eye_val)) if pd.notna(eye_val) else -1
                label_vector = row[self.target_cols].to_numpy(dtype=np.float32)
                demographic_values = {col: row[col] if pd.notna(row[col]) else None for col in self.demo_cols}
                
                self.samples.append({
                    "patient_id": str(row["patient_id"]),
                    "image_id": str(row["image_id"]), # Saved explicitly for logging
                    "image_paths": [p], # Passed as list to reuse vision logic
                    "eye_ids": [eye_id],
                    "label_vector": label_vector,
                    "demographics": demographic_values
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> HeteroData:
        sample = self.samples[index]
        data = HeteroData()
        
        # 1. Global Target Assignment
        # Because 1 sample = 1 graph (whether it contains 1 patient or 1 image), 
        # we always unsqueeze to shape [1, Num_Labels] so PyG batches it into [Batch_Size, Num_Labels]
        data.y = torch.tensor(sample["label_vector"], dtype=torch.float32).unsqueeze(0)
        
        # 2. Setup Base Nodes
        num_labels = len(self.target_cols)
        num_images = len(sample["image_paths"])
        
        data["patient"].x = torch.zeros((1, 1), dtype=torch.float32)
        data["label"].x = torch.zeros((num_labels, 1), dtype=torch.float32)
        data["label"].label_idx = torch.arange(num_labels, dtype=torch.long)
        
        if self.prediction_level == "central_hub":
            data["central"].x = torch.zeros((1, 1), dtype=torch.float32)
        
        # 3. Setup Demographic Nodes
        self._attach_demographics(data, sample["demographics"])
        
        # 4. Setup Image & Patch Nodes with Fast Edge Connections
        self._attach_vision_nodes(data, sample)
        
        # 5. Setup Structural Edges
        self._attach_structural_edges(data, num_labels, num_images)
        
        return data

    def _attach_demographics(self, data: HeteroData, demo_dict: Dict) -> None:
        num_demo = len(self.demo_specs)
        field_idx = torch.arange(num_demo, dtype=torch.long)
        kind = torch.tensor([0 if s.kind == "numeric" else 1 for s in self.demo_specs], dtype=torch.long)
        num_value = torch.zeros((num_demo, 1), dtype=torch.float32)
        cat_value = torch.zeros((num_demo,), dtype=torch.long)
        missing = torch.zeros((num_demo, 1), dtype=torch.float32)

        for i, spec in enumerate(self.demo_specs):
            value = demo_dict.get(spec.name, None)
            
            if spec.kind == "numeric":
                num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.isna(num):
                    missing[i, 0] = 1.0
                else:
                    num_value[i, 0] = (float(num) - spec.mean) / spec.std
            else:
                token = "__UNK__" if value is None or pd.isna(value) else str(value)
                cat_value[i] = spec.vocab.get(token, 0) if spec.vocab else 0
                if token == "__UNK__":
                    missing[i, 0] = 1.0

        data["demographic"].x = torch.zeros((num_demo, 1), dtype=torch.float32)
        data["demographic"].field_idx = field_idx
        data["demographic"].kind = kind
        data["demographic"].num_value = num_value
        data["demographic"].cat_value = cat_value
        data["demographic"].missing = missing

    def _attach_vision_nodes(self, data: HeteroData, sample: Dict) -> None:
        num_images = len(sample["image_paths"])
        
        # 1. Base Image Nodes
        data["image"].x = torch.zeros((num_images, 1), dtype=torch.float32)
        data["image"].image_idx = torch.arange(num_images, dtype=torch.long)
        data["image"].eye_idx = torch.tensor(
            [0 if v in (-1, None) else max(0, int(v) - 1) for v in sample["eye_ids"]],
            dtype=torch.long
        )
        
        # --- RESNET INTEGRATION: LOAD RAW IMAGES ---
        image_tensors = []
        for path in sample["image_paths"]:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img) # MUST include Resize((256, 256))
                if not isinstance(img, torch.Tensor):
                    img = transforms.ToTensor()(img)
            image_tensors.append(img)
            
        # Store the stacked images in the graph object!
        # PyG will automatically batch these into [Batch_Size_Images, 3, 256, 256]
        data["image"].raw_images = torch.stack(image_tensors, dim=0)
        
        # --- RESNET INTEGRATION: SETUP EMPTY PATCH NODES ---
        # We need 64 patches per image. We create an empty tensor so PyG doesn't crash 
        # when validating the edge connections. The ResNet will populate this later.
        num_patches = num_images * self.n_patches 
        data["patch"].x = torch.empty((num_patches, 0), dtype=torch.float32)
        
        
        # 2. FAST EDGE CONNECTIONS (Using your cached grid edges)
        p_adj_src, p_adj_dst = [], []
        p2i_src, p2i_dst = [], []
        i2p_src, i2p_dst = [], []
        i2pat_src, i2pat_dst = [], []
        pat2i_src, pat2i_dst = [], []
        
        for img_idx in range(num_images):
            patch_offset = img_idx * self.n_patches
            
            p_adj_src.append(self.base_patch_adj_src + patch_offset)
            p_adj_dst.append(self.base_patch_adj_dst + patch_offset)
            
            shifted_p2i_src = self.base_patch_to_img_src + patch_offset
            shifted_p2i_dst = self.base_patch_to_img_dst + img_idx
            
            p2i_src.append(shifted_p2i_src)
            p2i_dst.append(shifted_p2i_dst)
            i2p_src.append(shifted_p2i_dst)
            i2p_dst.append(shifted_p2i_src)
            
            i2pat_src.append(torch.tensor([img_idx], dtype=torch.long))
            i2pat_dst.append(torch.tensor([0], dtype=torch.long))
            pat2i_src.append(torch.tensor([0], dtype=torch.long))
            pat2i_dst.append(torch.tensor([img_idx], dtype=torch.long))
            
        data[("patch", "adjacent", "patch")].edge_index = torch.stack([torch.cat(p_adj_src), torch.cat(p_adj_dst)], dim=0)
        data[("patch", "adjacent", "patch")].edge_attr = self.base_patch_adj_attr.repeat(num_images, 1)
        
        data[("patch", "to", "image")].edge_index = torch.stack([torch.cat(p2i_src), torch.cat(p2i_dst)], dim=0)
        data[("patch", "to", "image")].edge_attr = torch.rand((data[("patch", "to", "image")].edge_index.size(1), 1), dtype=torch.float32)
        
        data[("image", "to", "patch")].edge_index = torch.stack([torch.cat(i2p_src), torch.cat(i2p_dst)], dim=0)
        data[("image", "to", "patch")].edge_attr = torch.rand((data[("image", "to", "patch")].edge_index.size(1), 1), dtype=torch.float32)
        
        data[("image", "to", "patient")].edge_index = torch.stack([torch.cat(i2pat_src), torch.cat(i2pat_dst)], dim=0)
        data[("image", "to", "patient")].edge_attr = torch.rand((data[("image", "to", "patient")].edge_index.size(1), 1), dtype=torch.float32)
        
        data[("patient", "to", "image")].edge_index = torch.stack([torch.cat(pat2i_src), torch.cat(pat2i_dst)], dim=0)
        data[("patient", "to", "image")].edge_attr = torch.rand((data[("patient", "to", "image")].edge_index.size(1), 1), dtype=torch.float32)
            
    def _attach_structural_edges(self, data: HeteroData, num_labels: int, num_images: int) -> None:
        num_demo = len(self.demo_specs)
        
        # User requested fully connected demographic nodes via correlation!
        demo_src, demo_dst, demo_attr = [], [], []
        for i in range(num_demo):
            for j in range(num_demo):
                if i != j:
                    demo_src.append(i)
                    demo_dst.append(j)
                    demo_attr.append([self.demo_corr_matrix[i, j].item()])
                    
        if len(demo_src) > 0:
            data[("demographic", "correlates", "demographic")].edge_index = torch.tensor([demo_src, demo_dst], dtype=torch.long)
            data[("demographic", "correlates", "demographic")].edge_attr = torch.tensor(demo_attr, dtype=torch.float32)
        else:
            data[("demographic", "correlates", "demographic")].edge_index = torch.empty((2, 0), dtype=torch.long)
            data[("demographic", "correlates", "demographic")].edge_attr = torch.empty((0, 1), dtype=torch.float32)
        
        data[("demographic", "to", "patient")].edge_index = torch.tensor([list(range(num_demo)), [0] * num_demo], dtype=torch.long)
        data[("demographic", "to", "patient")].edge_attr = torch.rand((num_demo, 1), dtype=torch.float32)
        
        data[("patient", "to", "demographic")].edge_index = torch.tensor([[0] * num_demo, list(range(num_demo))], dtype=torch.long)
        data[("patient", "to", "demographic")].edge_attr = torch.rand((num_demo, 1), dtype=torch.float32)
        
        corr_src, corr_dst, corr_attr = [], [], []
        for i in range(num_labels):
            for j in range(num_labels):
                if i != j:
                    corr_src.append(i)
                    corr_dst.append(j)
                    corr_attr.append([self.label_corr_matrix[i, j].item()])
                    
        data[("label", "correlates", "label")].edge_index = torch.tensor([corr_src, corr_dst], dtype=torch.long)
        data[("label", "correlates", "label")].edge_attr = torch.tensor(corr_attr, dtype=torch.float32)
        
        if self.prediction_level in ["patient", "dual_hub"]:
            data[("patient", "to", "label")].edge_index = torch.tensor([[0] * num_labels, list(range(num_labels))], dtype=torch.long)
            data[("patient", "to", "label")].edge_attr = torch.rand((num_labels, 1), dtype=torch.float32)
            
            data[("label", "to", "patient")].edge_index = torch.tensor([list(range(num_labels)), [0] * num_labels], dtype=torch.long)
            data[("label", "to", "patient")].edge_attr = torch.rand((num_labels, 1), dtype=torch.float32)
            
        if self.prediction_level in ["image", "dual_hub"]:
            img_src, lbl_dst = [], []
            for i in range(num_images):
                img_src.extend([i] * num_labels)
                lbl_dst.extend(list(range(num_labels)))
                
            data[("image", "to", "label")].edge_index = torch.tensor([img_src, lbl_dst], dtype=torch.long)
            data[("image", "to", "label")].edge_attr = torch.rand((len(img_src), 1), dtype=torch.float32)
            
            data[("label", "to", "image")].edge_index = torch.tensor([lbl_dst, img_src], dtype=torch.long)
            data[("label", "to", "image")].edge_attr = torch.rand((len(lbl_dst), 1), dtype=torch.float32)
            
        if self.prediction_level == "central_hub":
            # Image <-> Central
            data[("image", "to", "central")].edge_index = torch.tensor([list(range(num_images)), [0] * num_images], dtype=torch.long)
            data[("image", "to", "central")].edge_attr = torch.rand((num_images, 1), dtype=torch.float32)
            data[("central", "to", "image")].edge_index = torch.tensor([[0] * num_images, list(range(num_images))], dtype=torch.long)
            data[("central", "to", "image")].edge_attr = torch.rand((num_images, 1), dtype=torch.float32)
            
            # Patient <-> Central
            data[("patient", "to", "central")].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            data[("patient", "to", "central")].edge_attr = torch.rand((1, 1), dtype=torch.float32)
            data[("central", "to", "patient")].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            data[("central", "to", "patient")].edge_attr = torch.rand((1, 1), dtype=torch.float32)
            
            # Label <-> Central
            data[("label", "to", "central")].edge_index = torch.tensor([list(range(num_labels)), [0] * num_labels], dtype=torch.long)
            data[("label", "to", "central")].edge_attr = torch.rand((num_labels, 1), dtype=torch.float32)
            data[("central", "to", "label")].edge_index = torch.tensor([[0] * num_labels, list(range(num_labels))], dtype=torch.long)
            data[("central", "to", "label")].edge_attr = torch.rand((num_labels, 1), dtype=torch.float32)