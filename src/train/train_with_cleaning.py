"""
src/train/train_with_cleaning.py
添加 tqdm 进度条
"""

from __future__ import annotations
import os, random
from typing import Set

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 项目内部
from src.data.prepare_split import create_splits
from src.data.build_clips      import build_clips
from src.data.dataset_clip     import ClipDataset
from src.models.full_model     import ConfocalModel
from src.noise.vog_utils       import GradientVarianceTracker
from src.active.active_sampler import ActiveSampler
from src.utils.misc            import load_config, set_seed
from src.utils                 import metrics


def coteaching_one_epoch(
    model_a: nn.Module,
    model_b: nn.Module,
    loader,            # 可以是 DataLoader 或 tqdm(loader)
    opt_a, opt_b,
    crit,
    gv_a: GradientVarianceTracker,
    gv_b: GradientVarianceTracker,
    mix_ratio: float,
    drop_frac: float,
    device: torch.device,
) -> None:
    model_a.train(); model_b.train()

    for b_idx, (clips, noisy_y, _) in enumerate(loader):
        clips, noisy_y = clips.to(device), noisy_y.to(device)
        B = clips.size(0)

        logit_a, _ = model_a(clips)
        logit_b, _ = model_b(clips)
        loss_a = crit(logit_a, noisy_y)
        loss_b = crit(logit_b, noisy_y)

        start = b_idx * B
        idxs  = list(range(start, start + B))
        var_a = gv_a.get_variances().to(device)[idxs]
        var_b = gv_b.get_variances().to(device)[idxs]

        def select_idx(loss_vec, var_vec):
            rl = loss_vec.argsort().argsort().float()
            rv = var_vec.argsort().argsort().float()
            score = mix_ratio * rl + (1 - mix_ratio) * rv
            keep = int((1 - drop_frac) * len(score))
            return score.topk(keep, largest=False).indices

        sel_a = select_idx(loss_a, var_a)
        sel_b = select_idx(loss_b, var_b)

        if len(sel_b):
            opt_a.zero_grad()
            crit(logit_a[sel_b], noisy_y[sel_b]).mean().backward()
            opt_a.step()
        if len(sel_a):
            opt_b.zero_grad()
            crit(logit_b[sel_a], noisy_y[sel_a]).mean().backward()
            opt_b.step()

        with torch.no_grad():
            pa = torch.softmax(logit_a, dim=1)
            pb = torch.softmax(logit_b, dim=1)
            Wa, Wb = model_a.fc.weight.data, model_b.fc.weight.data
            ga, gb = [], []
            for i_loc, idx_gl in enumerate(idxs):
                one = torch.zeros_like(pa[i_loc]); one[noisy_y[i_loc]] = 1
                ga.append((Wa.t() @ (pa[i_loc] - one)).cpu())
                gb.append((Wb.t() @ (pb[i_loc] - one)).cpu())
            gv_a.update(idxs, torch.stack(ga))
            gv_b.update(idxs, torch.stack(gb))


def train_with_cleaning(cfg_path: str = "src/configs/confocal.yaml") -> None:
    cfg = load_config(cfg_path)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["output"]["save_dir"], exist_ok=True)

    # 数据准备
    splits = create_splits(cfg)
    classes = splits["classes"]
    train_pairs = splits["train"]
    test_pairs  = splits["test"]
    original    = splits["original_labels"]

    train_clips = build_clips(train_pairs, T=cfg["data"]["frames_per_clip"])
    test_clips  = build_clips(test_pairs,  T=cfg["data"]["frames_per_clip"])

    ds_train = ClipDataset(train_clips, original, mode="train", target_size=tuple(cfg["data"].get("target_size", [224,224])))
    ds_test  = ClipDataset(test_clips,  mode="test",  target_size=tuple(cfg["data"].get("target_size", [224,224])))

    dl_train = DataLoader(ds_train, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=0)
    dl_test  = DataLoader(ds_test,  batch_size=cfg["training"]["batch_size"], num_workers=0)

    n_train = len(train_clips)
    lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg["training"]["weight_decay"])

    # 模型与优化器
    model_a = ConfocalModel(cfg["model"]["num_classes"]).to(device)
    model_b = ConfocalModel(cfg["model"]["num_classes"]).to(device)
    opt_a   = torch.optim.Adam(model_a.parameters(), lr=lr, weight_decay=wd)
    opt_b   = torch.optim.Adam(model_b.parameters(), lr=lr, weight_decay=wd)
    crit    = nn.CrossEntropyLoss(reduction="none")
    gv_a = GradientVarianceTracker(model_a.feature_dim, n_train)
    gv_b = GradientVarianceTracker(model_b.feature_dim, n_train)

    # ① Co-Teaching + tqdm 进度条
    for ep in range(cfg["training"]["epochs"]):
        print(f"\n=== Epoch {ep+1}/{cfg['training']['epochs']} ===")
        tqdm_loader = tqdm(dl_train, desc=f"Epoch {ep+1}", leave=False)
        coteaching_one_epoch(
            model_a, model_b, tqdm_loader, opt_a, opt_b, crit,
            gv_a, gv_b,
            cfg["training"]["mix_ratio"],
            cfg["training"]["drop_fraction"],
            device
        )

    final_model = model_a
    sampler = ActiveSampler(cfg["active"]["method"])
    cleaned: Set[int] = set()

    # ② 主动清洗
    for rd in range(cfg["active"]["rounds"]):
        pool = [i for i in range(n_train) if i not in cleaned]
        if not pool:
            break
        k = max(1, int(cfg["active"]["query_ratio"] * len(pool)))
        sel = sampler.select(final_model, ds_train, pool, k, device)
        for idx in sel:
            true_lbl = original[str(idx)]
            frames, _ = ds_train.clip_list[idx]
            ds_train.clip_list[idx] = (frames, true_lbl)
            cleaned.add(idx)
        print(f"[Clean {rd+1}] 修正 {len(sel)} 个样本")

        ft_opt = torch.optim.Adam(final_model.parameters(), lr=lr*0.5, weight_decay=wd)
        final_model.train()
        for _ in range(max(1, cfg["training"]["epochs"] // 2)):
            for clips, y, _ in dl_train:
                ft_opt.zero_grad()
                loss = nn.CrossEntropyLoss()(final_model(clips.to(device))[0], y.to(device))
                loss.backward(); ft_opt.step()

    # ③ 测试评估
    final_model.eval()
    preds, gts = [], []
    for clips, y in tqdm(dl_test, desc="Testing"):
        logits, _ = final_model(clips.to(device))
        preds.extend(logits.argmax(1).cpu().tolist())
        gts.extend(y.tolist())

    acc = metrics.accuracy(preds, gts)
    mf1 = metrics.macro_f1(preds, gts)
    cm  = metrics.confusion_matrix(preds, gts, labels=list(range(len(classes))))
    metrics.plot_confusion_matrix(cm, classes, cfg["output"]["confusion_matrix_png"])
    torch.save(final_model.state_dict(), cfg["output"]["model_path"])
    print(f"\nTest ACC={acc:.4f} | Macro-F1={mf1:.4f}")
    print("模型已保存:", cfg["output"]["model_path"])


if __name__ == "__main__":
    train_with_cleaning()
