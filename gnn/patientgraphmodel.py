import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import GATConv, HeteroConv
except ImportError as exc:
    raise SystemExit("Missing dependency: torch-geometric. Install it before using this model.") from exc


@dataclass
class DemographicSpec:
    name: str
    kind: str  # numeric or categorical
    mean: float = 0.0
    std: float = 1.0
    vocab: Optional[Dict[str, int]] = None


class DensePatchAttention(nn.Module):
    """
    Dense Multi-Head Self-Attention over image patches using PyTorch scaled_dot_product_attention (SDPA).
    Incorporates 2D Gaussian spatial distance matrix as an additive attention bias.
    Executes via FlashAttention / cuDNN tensor cores per image (N=256) instead of irregular sparse GAT message passing,
    providing massive speedups and completely avoiding PyG graph edge explosion.
    """
    def __init__(
        self,
        hidden_dim: int,
        heads: int = 2,
        dropout: float = 0.1,
        grid_h: int = 16,
        grid_w: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads if hidden_dim % heads == 0 else hidden_dim
        self.inner_dim = self.head_dim * heads
        self.dropout = dropout
        self.n_patches = grid_h * grid_w
        
        self.q_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Precompute static 2D Gaussian distance matrix: [1, 1, N, N]
        coords_y = torch.arange(grid_h).repeat_interleave(grid_w)
        coords_x = torch.arange(grid_w).repeat(grid_h)
        dist_sq = (coords_y.unsqueeze(1) - coords_y.unsqueeze(0))**2 + (coords_x.unsqueeze(1) - coords_x.unsqueeze(0))**2
        sigma_sq = (grid_w / 2.0)**2
        gaussian_bias = torch.exp(-dist_sq.float() / sigma_sq).unsqueeze(0).unsqueeze(0)
        self.register_buffer("gaussian_bias", gaussian_bias)
        
        # Learnable per-head spatial bias scale (initialized to 1.0)
        self.bias_scale = nn.Parameter(torch.ones(1, heads, 1, 1))

    def forward(self, patch_x: torch.Tensor, num_images: int) -> torch.Tensor:
        """
        patch_x: [num_images * n_patches, hidden_dim]
        num_images: number of images in the batch
        Returns updated patch_x: [num_images * n_patches, hidden_dim]
        """
        b = num_images
        n = self.n_patches
        d = self.hidden_dim
        
        x_dense = patch_x.view(b, n, d)
        
        # Multi-head projections: [B, heads, N, head_dim]
        q = self.q_proj(x_dense).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_dense).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_dense).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        
        # Additive Gaussian spatial bias
        attn_mask = (self.bias_scale * self.gaussian_bias).to(dtype=q.dtype)
        
        # FlashAttention / cuDNN accelerated SDPA
        drop_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_p)
        
        # Reshape and project back: [B, N, D]
        attn_out = attn_out.transpose(1, 2).reshape(b, n, self.inner_dim)
        attn_out = self.out_proj(attn_out)
        attn_out = F.dropout(attn_out, p=drop_p, training=self.training)
        
        # Residual connection + LayerNorm
        x_dense = self.norm(x_dense + attn_out)
        
        return x_dense.view(-1, d)


class PatientGraphModel(nn.Module):
    """Heterogeneous Graph Neural Network for Patient-Level or Image-Level Predictions."""
    def __init__(
        self,
        num_labels: int,
        demographic_specs: Sequence[DemographicSpec],
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_images_per_patient: int = 32,
        heads: int = 2,
        prediction_level: str = "patient",
        initial_biases: Optional[Sequence[float]] = None,
        use_context: bool = True,
        use_image_features: bool = True,
        backbone: str = "retfound",
        patch_connectivity: Any = "fc",
        image_size: int = 512,
        patch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        self.backbone = backbone
        if prediction_level not in ["patient", "image", "dual_hub", "central_hub"]:
            raise ValueError("prediction_level must be either 'patient', 'image', 'dual_hub', or 'central_hub'")
            
        self.prediction_level = prediction_level
        self.use_context = use_context
        self.use_image_features = use_image_features
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.demographic_specs = list(demographic_specs)
        self.dropout = dropout
        self.max_images_per_patient = max_images_per_patient
        
        self.patch_connectivity = str(patch_connectivity).lower()
        self.image_size = image_size
        self.patch_size = (16 if backbone == "retfound" else 32) if patch_size is None else patch_size
        self.grid_h = self.image_size // self.patch_size
        self.grid_w = self.image_size // self.patch_size
        self.n_patches = self.grid_h * self.grid_w
        
        self.use_sdpa_patch = self.use_image_features and (self.patch_connectivity in ("fc", "sdpa", "dense", "transformer"))
        
        if self.use_sdpa_patch:
            self.patch_attns = nn.ModuleList([
                DensePatchAttention(
                    hidden_dim=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    grid_h=self.grid_h,
                    grid_w=self.grid_w,
                ) for _ in range(num_layers)
            ])
        else:
            self.patch_attns = None

        # --- Load Pre-trained Vision Backbone ---
        try:
            import timm
        except ImportError as exc:
            raise SystemExit("Missing dependency: timm. Install it via 'pip install timm'") from exc

        if self.backbone == "retfound":
            self.cnn = timm.create_model("hf_hub:bitfount/RETFound_MAE", pretrained=True, dynamic_img_size=True)
            for param in self.cnn.parameters():
                param.requires_grad = False
            cnn_out_dim = 1024
        elif self.backbone == "resnet":
            self.cnn = timm.create_model("resnet50", pretrained=True, features_only=False)
            cnn_out_dim = 2048
        else:
            raise ValueError("backbone must be 'retfound' or 'resnet'")
        
        # 2. Dynamic Stepped Compression Encoders
        def build_stepped_encoder(in_dim: int, out_dim: int, drop_rate: float) -> nn.Sequential:
            layers = []
            curr_dim = in_dim
            while curr_dim > out_dim * 2:
                next_dim = curr_dim // 2
                layers.extend([
                    nn.Linear(curr_dim, next_dim),
                    nn.LayerNorm(next_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                ])
                curr_dim = next_dim
            
            # Final step to the target hidden dimension
            layers.extend([
                nn.Linear(curr_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
            ])
            return nn.Sequential(*layers)
            
        self.patch_encoder = build_stepped_encoder(cnn_out_dim, hidden_dim, dropout)
        self.cls_encoder = build_stepped_encoder(cnn_out_dim, hidden_dim, dropout)

        # --- Demographic Encoders ---
        self.demo_field_embed = nn.Embedding(len(self.demographic_specs), hidden_dim)
        self.demo_kind_embed = nn.Embedding(2, hidden_dim)
        self.demo_numeric = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.demo_cat_tables = nn.ModuleList()
        self.cat_field_to_table: Dict[int, int] = {}
        cat_count = 0
        for field_idx, spec in enumerate(self.demographic_specs):
            if spec.kind == "categorical":
                size = len(spec.vocab or {"__UNK__": 0})
                self.demo_cat_tables.append(nn.Embedding(size, hidden_dim))
                self.cat_field_to_table[field_idx] = cat_count
                cat_count += 1

        self.image_idx_embed = nn.Embedding(max_images_per_patient + 1, hidden_dim)
        self.eye_idx_embed = nn.Embedding(4, hidden_dim)
        self.patient_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.label_token = nn.Embedding(num_labels, hidden_dim)

        self.type_bias = nn.ParameterDict(
            {
                "patch": nn.Parameter(torch.zeros(hidden_dim)),
                "image": nn.Parameter(torch.zeros(hidden_dim)),
                "demographic": nn.Parameter(torch.zeros(hidden_dim)),
                "patient": nn.Parameter(torch.zeros(hidden_dim)),
                "label": nn.Parameter(torch.zeros(hidden_dim)),
                "central": nn.Parameter(torch.zeros(hidden_dim)),
            }
        )
        
        if prediction_level == "central_hub":
            self.central_token = nn.Parameter(torch.zeros(1, hidden_dim))

        # --- Heterogeneous Convolutions ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            
            # Base structural pathways
            conv_dict = {
                ("label", "correlates", "label"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=True, edge_dim=1),
            }
            if self.use_image_features:
                if not self.use_sdpa_patch:
                    conv_dict[("patch", "adjacent", "patch")] = GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=True, edge_dim=1)
                conv_dict.update({
                    ("patch", "to", "image"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1),
                    ("image", "to", "patch"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1),
                })
            
            # Context Pathways (removed for image-only baselines)
            if self.use_context:
                conv_dict.update({
                    ("image", "to", "patient"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1),
                    ("patient", "to", "image"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1),
                    ("demographic", "to", "patient"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1),
                    ("patient", "to", "demographic"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1),
                    ("demographic", "correlates", "demographic"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=True, edge_dim=1),
                })
            
            # --- DYNAMIC LABEL PATHWAYS ---
            if self.prediction_level in ["patient", "dual_hub"] and self.use_context:
                conv_dict[("patient", "to", "label")] = GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1)
                conv_dict[("label", "to", "patient")] = GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1)
            
            if self.prediction_level in ["image", "dual_hub"]:
                conv_dict[("image", "to", "label")] = GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1)
                conv_dict[("label", "to", "image")] = GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1)
                
            if self.prediction_level == "central_hub":
                conv_dict[("image", "to", "central")] = GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1)
                conv_dict[("central", "to", "image")] = GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1)
                
                conv_dict[("patient", "to", "central")] = GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1)
                conv_dict[("central", "to", "patient")] = GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1)
                
                conv_dict[("label", "to", "central")] = GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1)
                conv_dict[("central", "to", "label")] = GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False, edge_dim=1)

            conv = HeteroConv(conv_dict, aggr="mean")
            self.convs.append(conv)
            
            self.norms.append(
                nn.ModuleDict(
                    {
                        "patch": nn.LayerNorm(hidden_dim),
                        "image": nn.LayerNorm(hidden_dim),
                        "demographic": nn.LayerNorm(hidden_dim),
                        "patient": nn.LayerNorm(hidden_dim),
                        "label": nn.LayerNorm(hidden_dim),
                        "central": nn.LayerNorm(hidden_dim),
                    }
                )
            )

        self.label_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        self.image_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels, bias=False),
        )
        
        self.central_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels, bias=False),
        )

        if initial_biases is not None:
            bias_tensor = torch.tensor(initial_biases, dtype=torch.float32)
        else:
            bias_tensor = torch.zeros(num_labels, dtype=torch.float32)

        self.output_bias = nn.Parameter(bias_tensor)

    def encode_demographics(self, data: HeteroData) -> torch.Tensor:
        field_idx = data["demographic"].field_idx.long()
        kind = data["demographic"].kind.long()
        num_value = data["demographic"].num_value.float()
        cat_value = data["demographic"].cat_value.long()
        missing = data["demographic"].missing.float()

        out = torch.zeros((field_idx.size(0), self.hidden_dim), device=field_idx.device)

        for i, spec in enumerate(self.demographic_specs):
            mask = field_idx == i
            if not mask.any():
                continue

            base = self.demo_field_embed(field_idx[mask]) + self.demo_kind_embed(kind[mask])
            if spec.kind == "numeric":
                emb = self.demo_numeric(torch.cat([num_value[mask], missing[mask]], dim=-1))
            else:
                emb = self.demo_cat_tables[self.cat_field_to_table[i]](cat_value[mask])
            out[mask] = base + emb

        return out + self.type_bias["demographic"]

    def encode_images(self, data: HeteroData) -> torch.Tensor:
        image_idx = data["image"].image_idx.long().clamp(min=0, max=self.max_images_per_patient)
        eye_idx = data["image"].eye_idx.long().clamp(min=0, max=3)
        return self.image_idx_embed(image_idx) + self.eye_idx_embed(eye_idx) + self.type_bias["image"]

    def forward(self, data: HeteroData) -> torch.Tensor:
        
        # --- GNN MESSAGE PASSING ---
        # 1. Standard GNN Dictionary Setup
        x_dict = {
            "image": self.encode_images(data),
            "demographic": self.encode_demographics(data),
            "patient": self.patient_token.expand(data["patient"].num_nodes, -1) + self.type_bias["patient"],
            "label": self.label_token(data["label"].label_idx.long()) + self.type_bias["label"],
        }
        
        if self.prediction_level == "central_hub":
            x_dict["central"] = self.central_token.expand(data["central"].num_nodes, -1) + self.type_bias["central"]
        
        # --- RETFOUND FEATURE EXTRACTION ---
        if self.use_image_features:
            # 2. Get the batched raw images from PyG: [Total_Images_in_Batch, 3, 256, 256]
            imgs = data["image"].raw_images 
            
            # 3. Extract Deep Spatial Features
            if self.backbone == "retfound":
                with torch.no_grad():
                    features = self.cnn.forward_features(imgs)
                    
                if features.dim() == 3:
                    expected_patches = (imgs.shape[2] // 16) * (imgs.shape[3] // 16)
                    if features.shape[1] > expected_patches:
                        # Extract the global CLS token and inject it into the Image Node!
                        cls_token = features[:, 0, :]
                        x_dict["image"] = x_dict["image"] + self.cls_encoder(cls_token)
                        patch_features_seq = features[:, 1:, :] 
                    else:
                        patch_features_seq = features
                        
                    b, num_patches, c = patch_features_seq.shape
                    patch_features = patch_features_seq.reshape(-1, c)
                else:
                    b, c, h, w = features.shape
                    global_pooled = features.mean(dim=(-2, -1))
                    x_dict["image"] = x_dict["image"] + self.cls_encoder(global_pooled)
                    patch_features = features.permute(0, 2, 3, 1).reshape(-1, c)
                    
            elif self.backbone == "resnet":
                features = self.cnn.forward_features(imgs)
                b, c, h, w = features.shape
                # Global Average Pooling (GAP) -> Inject into Image Hub Node
                global_pooled = features.mean(dim=(-2, -1))
                x_dict["image"] = x_dict["image"] + self.cls_encoder(global_pooled)
                patch_features = features.permute(0, 2, 3, 1).reshape(-1, c)
                
            x_dict["patch"] = self.patch_encoder(patch_features) + self.type_bias["patch"]

        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # 1. Intra-image dense patch self-attention (FlashAttention SDPA with Gaussian spatial bias)
            if self.use_image_features and self.patch_attns is not None:
                num_images = data["image"].num_nodes
                x_dict["patch"] = self.patch_attns[layer_idx](x_dict["patch"], num_images=num_images)

            # 2. Heterogeneous bipartite message passing (patch <-> image, image <-> label, etc.)
            out_dict = conv(x_dict, data.edge_index_dict, data.edge_attr_dict)
            next_x = {}
            for node_type, x in x_dict.items():
                if node_type in out_dict:
                    h = F.relu(out_dict[node_type])
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    next_x[node_type] = norm[node_type](x + h)
                else:
                    next_x[node_type] = x
            x_dict = next_x

        # --- DYNAMIC OUTPUT ROUTING ---
        if self.prediction_level in ["patient", "dual_hub"]:
            # Read from the label nodes (which aggregate both hubs simultaneously in dual_hub)
            logits = self.label_head(x_dict["label"]).squeeze(-1)
            num_targets = data["patient"].num_nodes if "patient" in data else data["image"].num_nodes
            logits = logits.view(num_targets, self.num_labels)
            return logits + self.output_bias
            
        elif self.prediction_level == "image":
            # Read directly from the enriched image nodes
            # Output inherently matches shape [num_images, num_labels]
            logits = self.image_head(x_dict["image"])
            return logits + self.output_bias
            
        elif self.prediction_level == "central_hub":
            logits = self.central_head(x_dict["central"])
            num_targets = data["patient"].num_nodes if "patient" in data else data["image"].num_nodes
            logits = logits.view(num_targets, self.num_labels)
            return logits + self.output_bias