# src/models/resnet_backbone.py
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Backbone, self).__init__()
        # 加载预训练的 ResNet18 模型
        resnet = models.resnet18(pretrained=pretrained)
        # 移除最后的FC层，保留平均池化层使输出为512维特征
        self.features = nn.Sequential(*(list(resnet.children())[:-1]))

    def forward(self, x):
        """
        输入 x: 单帧图像张量，形状 [B, C, H, W]。
        输出特征: [B, 512] 张量。
        """
        out = self.features(x)  # [B, 512, 1, 1] after avgpool
        out = out.view(out.size(0), -1)  # 展平为 [B, 512]
        return out
