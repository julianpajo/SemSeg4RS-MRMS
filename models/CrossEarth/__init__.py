"""
models/CrossEarth
-----------------
CrossEarth: DINOv2 + Rein PEFT adapter for remote sensing semantic segmentation.

Paper:  "CrossEarth: Geospatial Vision Foundation Model for Domain
         Generalizable Remote Sensing Semantic Segmentation"
        TPAMI 2025 — https://arxiv.org/abs/2410.22629

Architecture:
  DINOv2 (torch.hub, FROZEN)  +  ReinAdapter (trainable)
      ↓  4 feature maps  (B, embed_dim, H/14, W/14)
  MLADecoder / LinearDecoder  (trainable)
      ↓
  logits  (B, num_classes, H, W)

What comes from libraries (zero code):
  - Full DINOv2 from torch.hub (facebookresearch/dinov2)

What is custom (~200 lines total):
  - ReinAdapter  (rein_adapter.py)  — PEFT adapter inspired by Rein CVPR2024
  - DINOv2WithRein  (dinov2_rein.py) — hooks + multi-scale reshape
  - MLADecoder / LinearDecoder  (decoder.py)

Dependencies:
  pip install torch torchvision

Backbone setup (one of the two):
  A) Automatic:
       model = CrossEarthSeg.from_pretrained("dinov2_vitl14_reg", num_classes=14)
  B) Local (download from https://dl.fbaipublicfiles.com/dinov2/):
       model = CrossEarthSeg(variant="dinov2_vitl14_reg", num_classes=14)
       model.load_backbone_checkpoint("checkpoints/dinov2_vitl14_pretrain.pth")

Input notes:
  H and W must be multiples of 14 (DINOv2 patch_size).
  Use 504×504 (36×14) or 518×518 (37×14) as crop/resize sizes.

Public API
----------
    CrossEarthSeg   – full model
    ReinAdapter     – standalone adapter
    DINOv2WithRein  – standalone backbone+rein
    MLADecoder      – standalone MLA decoder
    LinearDecoder   – standalone linear decoder
    DINOV2_CONFIGS  – DINOv2 variant configurations
"""

from .crossearth_seg import CrossEarthSeg
from .rein_adapter   import ReinAdapter
from .dinov2_rein    import DINOv2WithRein, DINOV2_CONFIGS
from .decoder        import MLADecoder, LinearDecoder

__all__ = [
    "CrossEarthSeg",
    "ReinAdapter",
    "DINOv2WithRein",
    "MLADecoder",
    "LinearDecoder",
    "DINOV2_CONFIGS",
]