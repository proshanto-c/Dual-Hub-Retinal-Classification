from pathlib import Path
from typing import Callable, List, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from gnn.patientgraphmodel import DemographicSpec

class BRSETFusionDataset(Dataset):
    """
    Standard PyTorch Multimodal Dataset for Fundus Photos + Tabular Clinical Data.
    Designed for ResNet-50 + Clinical MLP baseline without graph overhead.
    """
    def __init__(
        self,
        data: Union[Path, str, pd.DataFrame],
        image_dir: Path,
        target_cols: List[str],
        demo_cols: List[str],
        demo_specs: List[DemographicSpec],
        image_size: int = 512,
        transform: Optional[Callable] = None,
        validate_paths: bool = False,
        load_images: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(data, (Path, str)):
            self.frame = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.frame = data.copy()
        else:
            raise ValueError("data must be a path-like string/Path or a pandas DataFrame.")

        self.image_dir = Path(image_dir)
        self.target_cols = list(target_cols)
        self.demo_cols = list(demo_cols)
        self.demo_specs = list(demo_specs)
        self.image_size = image_size
        self.transform = transform
        self.load_images = load_images

        # Split demographic specs into continuous and categorical
        self.num_specs = [s for s in self.demo_specs if s.kind == "numeric"]
        self.cat_specs = [s for s in self.demo_specs if s.kind == "categorical"]

        # 1. Pre-vectorize continuous & missingness tensors
        n_samples = len(self.frame)
        n_num = len(self.num_specs)
        cont_mat = np.zeros((n_samples, n_num), dtype=np.float32)
        miss_mat = np.zeros((n_samples, n_num), dtype=np.float32)

        for i, s in enumerate(self.num_specs):
            if s.name in self.frame.columns:
                series = pd.to_numeric(self.frame[s.name], errors="coerce").to_numpy()
                isna = np.isnan(series)
                miss_mat[:, i] = isna.astype(np.float32)
                std = s.std if (s.std is not None and s.std > 1e-6) else 1.0
                mean = s.mean if s.mean is not None else 0.0
                norm_vals = np.where(isna, 0.0, (series - mean) / std)
                cont_mat[:, i] = norm_vals.astype(np.float32)
            else:
                miss_mat[:, i] = 1.0

        self.continuous_tensor = torch.from_numpy(cont_mat)
        self.missing_tensor = torch.from_numpy(miss_mat)

        # 2. Pre-vectorize categorical tensor
        n_cat = len(self.cat_specs)
        cat_mat = np.zeros((n_samples, n_cat), dtype=np.int64)

        for i, s in enumerate(self.cat_specs):
            vocab = s.vocab or {}
            if s.name in self.frame.columns:
                col_vals = self.frame[s.name].fillna("__UNK__").astype(str)
                cat_mat[:, i] = col_vals.map(lambda v: vocab.get(v, 0)).to_numpy(dtype=np.int64)
            else:
                cat_mat[:, i] = 0

        self.categorical_tensor = torch.from_numpy(cat_mat)

        # 3. Pre-vectorize target labels
        target_mat = np.zeros((n_samples, len(self.target_cols)), dtype=np.float32)
        for i, col in enumerate(self.target_cols):
            if col in self.frame.columns:
                target_mat[:, i] = pd.to_numeric(self.frame[col], errors="coerce").fillna(0.0).to_numpy()
        self.target_tensor = torch.from_numpy(target_mat)

        # 4. Resolve image paths
        self.image_paths: List[Path] = []
        self.image_ids: List[str] = []

        for _, row in self.frame.iterrows():
            img_id = str(row.get("image_id", ""))
            self.image_ids.append(img_id)

            raw_path = row.get("image_path", f"{img_id}.jpg")
            filename = Path(raw_path).name
            if self.image_dir and (self.image_dir / filename).is_file():
                p = self.image_dir / filename
            else:
                p = Path(raw_path) if Path(raw_path).is_absolute() else self.image_dir / filename

            if validate_paths and self.load_images and not p.is_file():
                raise FileNotFoundError(f"Missing fundus image: {p}")

            self.image_paths.append(p)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        if self.load_images:
            path = self.image_paths[index]
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                if not isinstance(img, torch.Tensor):
                    img = transforms.ToTensor()(img)
        else:
            img = torch.zeros((3, 1, 1), dtype=torch.float32)

        return {
            "images": img,
            "continuous": self.continuous_tensor[index],
            "missing": self.missing_tensor[index],
            "categorical": self.categorical_tensor[index],
            "targets": self.target_tensor[index],
            "image_id": self.image_ids[index],
        }
