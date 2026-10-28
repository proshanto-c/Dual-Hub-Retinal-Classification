import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Any
import timm

class TabularClinicalEncoder(nn.Module):
    """
    Standard tabular neural network for clinical features:
      - Categorical fields: Learns an entity embedding per categorical column
      - Numeric fields: Standardized continuous input + missingness flag
      - Dense MLP: Dense layers with LayerNorm, ReLU, and Dropout
    """
    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: List[int],
        out_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.15
    ):
        super().__init__()
        self.num_continuous = num_continuous
        
        # Categorical entity embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, max(4, min(32, (cardinality + 1) // 2)))
            for cardinality in cat_cardinalities
        ])
        
        total_cat_dim = sum(e.embedding_dim for e in self.cat_embeddings)
        in_dim = (num_continuous * 2) + total_cat_dim # continuous + missing flag + cat embeddings
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        continuous: torch.Tensor,
        missing: torch.Tensor,
        categorical: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        continuous: [Batch, num_continuous]
        missing:    [Batch, num_continuous]
        categorical: [Batch, num_categorical]
        """
        cat_reps = []
        if len(self.cat_embeddings) > 0 and categorical is not None and categorical.numel() > 0:
            for i, emb in enumerate(self.cat_embeddings):
                cat_reps.append(emb(categorical[:, i]))
            
        parts = [continuous, missing]
        if cat_reps:
            parts.extend(cat_reps)
            
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)

class ResNetClinicalFusion(nn.Module):
    """
    Standard multimodal late-fusion baseline:
      - Vision Branch: Pretrained ResNet-50 feature extractor -> Global Average Pooling -> Linear projection
      - Clinical Branch: Tabular Clinical Encoder on patient clinical context
      - Fusion: Concatenation [h_img || h_clin] -> MLP Fusion Layer
      - Classification Head: Multi-label logits output
    """
    def __init__(
        self,
        num_classes: int,
        num_continuous: int = 2,
        cat_cardinalities: Optional[List[int]] = None,
        fusion_dim: int = 512,
        dropout: float = 0.15,
        modality: str = "img_ctx", # "img_ctx", "image_only", or "ctx_only"
        pretrained: bool = True,
        initial_biases: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.modality = modality
        self.use_image = modality in ["img_ctx", "image_only"]
        self.use_context = modality in ["img_ctx", "ctx_only"]
        
        # 1. Vision Branch: ResNet-50 Backbone
        if self.use_image:
            self.backbone = timm.create_model("resnet50", pretrained=pretrained, num_classes=0) # Pooled 2048-dim
            self.img_proj = nn.Sequential(
                nn.Linear(2048, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        else:
            self.backbone = None
            self.img_proj = None

        # 2. Clinical Context Branch
        if self.use_context:
            cat_cards = cat_cardinalities or []
            self.clinical_encoder = TabularClinicalEncoder(
                num_continuous=num_continuous,
                cat_cardinalities=cat_cards,
                out_dim=fusion_dim // 2,
                hidden_dim=256,
                dropout=dropout
            )
        else:
            self.clinical_encoder = None

        # 3. Multimodal Fusion Head
        if self.use_image and self.use_context:
            total_in = fusion_dim + (fusion_dim // 2)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(total_in, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            head_in = fusion_dim
        elif self.use_image:
            self.fusion_mlp = nn.Identity()
            head_in = fusion_dim
        else:
            self.fusion_mlp = nn.Identity()
            head_in = fusion_dim // 2

        self.classifier = nn.Linear(head_in, num_classes)
        
        # Optional class prevalence bias initialization
        if initial_biases is not None:
            with torch.no_grad():
                bias_t = torch.as_tensor(initial_biases, dtype=torch.float32)
                self.classifier.bias.copy_(bias_t)

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        continuous: Optional[torch.Tensor] = None,
        missing: Optional[torch.Tensor] = None,
        categorical: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        images: [Batch, 3, H, W]
        continuous: [Batch, num_continuous]
        missing: [Batch, num_continuous]
        categorical: [Batch, num_categorical]
        """
        reps = []
        if self.use_image:
            if images is None:
                raise ValueError("images required for modality " + self.modality)
            # ResNet-50 feature extraction [Batch, 2048]
            feat = self.backbone(images)
            h_img = self.img_proj(feat)
            reps.append(h_img)

        if self.use_context:
            if continuous is None or categorical is None:
                raise ValueError("clinical inputs required for modality " + self.modality)
            h_clin = self.clinical_encoder(continuous, missing, categorical)
            reps.append(h_clin)

        if len(reps) == 2:
            fused = self.fusion_mlp(torch.cat(reps, dim=-1))
        else:
            fused = reps[0]

        logits = self.classifier(fused)
        return logits
