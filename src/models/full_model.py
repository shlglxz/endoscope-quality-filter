"""
src/models/full_model.py
ConfocalModel：ResNet18 + TemporalAttention
"""

import torch
import torch.nn as nn
from .resnet_backbone import ResNet18Backbone
from .attention_pooling import TemporalAttentionPool

class ConfocalModel(nn.Module):
    """
    Clip-level model: backbone (512-d each frame) → temporal attention pooling → classifier
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        # 每帧特征提取
        self.backbone = ResNet18Backbone(pretrained=pretrained)
        self.feature_dim = 512
        # 时间注意力池化
        self.pool = TemporalAttentionPool(self.feature_dim)
        # 分类头
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        clip: [B, T, C, H, W]
        returns: logits [B, num_classes], emb [B, feature_dim]
        """
        B, T, C, H, W = clip.shape
        feats = []
        for t in range(T):
            x_t = clip[:, t]                  # [B, C, H, W]
            f_t = self.backbone(x_t)          # [B, 512]
            feats.append(f_t.unsqueeze(1))    # [B,1,512]
        feats = torch.cat(feats, dim=1)       # [B, T, 512]
        emb   = self.pool(feats)              # [B, 512]
        logits = self.fc(emb)                 # [B, num_classes]
        return logits, emb
