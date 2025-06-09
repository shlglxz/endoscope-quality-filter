"""
src/data/dataset_clip.py
统一 Resize + ToTensor，保证每帧尺寸一致
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T


class ClipDataset(Dataset):
    def __init__(
            self,
            clip_list,
            original_labels=None,
            mode="train",
            target_size=None  # e.g. (224,224)
    ):
        """
        clip_list: list of ([frame_path,...], noisy_label)
        original_labels: dict {str(idx): true_label}
        mode: 'train' or 'test'
        target_size: (H,W) or None
        """
        self.clip_list = clip_list
        self.original_labels = original_labels or {}
        self.mode = mode

        # 构建 transform：Resize + ToTensor
        ops = []
        if target_size is not None:
            ops.append(T.Resize(target_size, interpolation=Image.BILINEAR))
        ops.append(T.ToTensor())  # outputs [C,H,W], values in [0,1]
        self.transform = T.Compose(ops)

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        frames, noisy_lbl = self.clip_list[idx]
        # 读取并变换每帧
        clips = []
        for fp in frames:
            img = Image.open(fp).convert("RGB")
            img = self.transform(img)
            clips.append(img)
        # 拼成 [T, C, H, W]
        clip_tensor = torch.stack(clips, dim=0)

        if self.mode == "train":
            # 返回 (clip, noisy_label, true_label)
            true_lbl = self.original_labels.get(str(idx), noisy_lbl)
            return clip_tensor, torch.tensor(noisy_lbl, dtype=torch.long), torch.tensor(true_lbl, dtype=torch.long)
        else:
            # 返回 (clip, label)
            return clip_tensor, torch.tensor(noisy_lbl, dtype=torch.long)
