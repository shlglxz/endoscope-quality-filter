"""
src/data/prepare_split.py
──────────────────────────
读取 CSV（三列：patient_name,label,image_name）
→ 生成帧绝对路径
→ 按患者 7:2:1 划分
→ 训练集随机翻转 10 % 标签 (对称噪声)
→ 返回 splits 字典：
   {
       "classes": [0,1],
       "train":   [[path,label], ...]  # 噪声标签
       "val":     [[path,label], ...]  # 干净
       "test":    [[path,label], ...]  # 干净
       "original_labels": {"0":0,"1":1,...}  # train 真标签索引
   }
"""
from __future__ import annotations
import os, random, json, pathlib
import pandas as pd


def create_splits(cfg: dict) -> dict:
    # 读取基础信息
    root_dir = pathlib.Path(cfg["data"]["root_dir"]) / "dataset"
    csv_path = pathlib.Path(cfg["data"]["csv_file"])
    noise_p  = cfg["data"]["noise_rate"]

    df = pd.read_csv(csv_path, encoding="utf-8")

    # 拼接完整帧路径
    df["full_path"] = df.apply(
        lambda r: str(root_dir / r.patient_name / r.image_name), axis=1
    )

    # -------- 1) 按患者 7:2:1 划分 --------
    patients = df.patient_name.unique().tolist()
    random.shuffle(patients)

    n_total = len(patients)
    n_train = int(0.7 * n_total)
    n_val   = int(0.2 * n_total)

    train_pat = set(patients[:n_train])
    val_pat   = set(patients[n_train:n_train+n_val])

    train_df = df[df.patient_name.isin(train_pat)].copy()
    val_df   = df[df.patient_name.isin(val_pat)].copy()
    test_df  = df[~df.patient_name.isin(train_pat | val_pat)].copy()

    classes = sorted(df.label.unique().tolist())         # e.g. [0,1]
    num_cls = len(classes)

    # -------- 2) 训练集注入对称随机噪声 --------
    noisy_pairs: list[list] = []
    original_labels: dict[str, int] = {}

    for idx, row in train_df.iterrows():
        orig_lbl = int(row.label)
        original_labels[str(len(noisy_pairs))] = orig_lbl   # 记录真标签

        if random.random() < noise_p:                       # 翻转
            cand = [c for c in classes if c != orig_lbl]
            noisy_lbl = random.choice(cand)
        else:
            noisy_lbl = orig_lbl

        noisy_pairs.append([row.full_path, noisy_lbl])

    # -------- 3) val / test 不加噪声 --------
    val_pairs  = val_df.apply(lambda r: [r.full_path, int(r.label)], axis=1).tolist()
    test_pairs = test_df.apply(lambda r: [r.full_path, int(r.label)], axis=1).tolist()

    splits = {
        "classes": classes,
        "train":   noisy_pairs,
        "val":     val_pairs,
        "test":    test_pairs,
        "original_labels": original_labels
    }

    # -------- 4) 保存 JSON 方便调试 --------
    out_dir = pathlib.Path(cfg["output"]["save_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "splits.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    print("saved", out_json)

    return splits
