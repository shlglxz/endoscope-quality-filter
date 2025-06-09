"""
杂项工具：读取 YAML 配置、设置随机种子等
"""

from __future__ import annotations
import os
import random
import yaml
import numpy as np
import torch


# ------------------------------------------------------------------
# 读取配置
# ------------------------------------------------------------------
def load_config(path: str) -> dict:
    """读取 YAML → dict"""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ------------------------------------------------------------------
# 固定随机种子，保证可复现
# ------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CuDNN 设置（完全可复现，但可能降低速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
