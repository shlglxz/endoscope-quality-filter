"""
度量指标与可视化工具
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix as _sk_confusion


# ------------------------------------------------------------------
# 基础指标
# ------------------------------------------------------------------
def accuracy(pred: list[int] | np.ndarray,
             true: list[int] | np.ndarray) -> float:
    """简单分类准确率"""
    pred = np.asarray(pred)
    true = np.asarray(true)
    return float((pred == true).mean())


def macro_f1(pred: list[int] | np.ndarray,
             true: list[int] | np.ndarray) -> float:
    """宏平均 F1"""
    pred = np.asarray(pred)
    true = np.asarray(true)
    return f1_score(true, pred, average="macro")


def confusion_matrix(pred: list[int] | np.ndarray,
                     true: list[int] | np.ndarray,
                     labels: list[int] | None = None) -> np.ndarray:
    """包装一下 sklearn.confusion_matrix，保证返回 np.ndarray"""
    return _sk_confusion(true, pred, labels=labels)


# ------------------------------------------------------------------
# 混淆矩阵可视化
# ------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray,
                          class_names: list[str] | list[int],
                          save_path: str,
                          normalize: bool = False,
                          figsize: tuple[int, int] = (5, 4)) -> None:
    """
    绘制并保存混淆矩阵；默认按行归一化关闭，可通过 normalize=True 开
    """
    if normalize:
        cm = cm.astype(float)
        row_sum = cm.sum(axis=1, keepdims=True) + 1e-8
        cm = cm / row_sum

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i,
                     format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    # 创建文件夹
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
