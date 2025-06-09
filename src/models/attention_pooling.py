# src/models/attention_pooling.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPool(nn.Module):
    def __init__(self, feature_dim):
        super(TemporalAttentionPool, self).__init__()
        # 用于计算注意力分数的线性层，将每帧的 feature_dim -> 1
        self.attn_fc = nn.Linear(feature_dim, 1)

    def forward(self, frame_features):
        """
        frame_features: 张量 [B, T, D]，B=batch_size, T=时间帧数, D=特征维度。
        返回聚合后的clip特征 [B, D]。
        """
        # 计算每帧的注意力分数 [B, T, 1]
        # 将frame_features展平成 [B*T, D] 再通过线性层，再reshape回 [B, T, 1]
        B, T, D = frame_features.size()
        scores = self.attn_fc(frame_features.view(B * T, D))  # [B*T, 1]
        scores = scores.view(B, T)  # [B, T]
        # 计算注意力权重，使用softmax归一化
        weights = F.softmax(scores, dim=1)  # [B, T]
        # 利用广播机制将权重乘以对应帧特征
        weighted_features = frame_features * weights.unsqueeze(-1)  # [B, T, D]
        # 沿时间轴求和，得到聚合的clip特征 [B, D]
        agg_features = weighted_features.sum(dim=1)
        return agg_features
