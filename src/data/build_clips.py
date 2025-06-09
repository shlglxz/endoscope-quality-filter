# src/data/build_clips.py
import pathlib


def build_clips(pair_list, T=8, stride=1):
    """
    pair_list: [(frame_path, label), …] —— 已经是具体帧
    连续 T 帧打包成一个 clip，同一个 patient 的帧才连在一起。
    """
    clips = []
    # 先按 patient / 帧序号排序
    def key(path):
        p = pathlib.Path(path)
        return (p.parent.name, p.name)  # (患者文件夹, 文件名)
    pair_list.sort(key=lambda x: key(x[0]))

    buf, cur_patient = [], None
    for path, label in pair_list:
        p = pathlib.Path(path).parent.name
        if cur_patient is None:
            cur_patient = p
        # 遇到新患者就清空缓冲
        if p != cur_patient:
            buf.clear()
            cur_patient = p
        buf.append((path, label))
        # 每 stride 滚动窗口
        if len(buf) >= T:
            window = buf[-T:]
            # 只要窗口里出现 0 就标 0，否则 1
            win_label = 0 if any(l==0 for _,l in window) else 1
            clips.append(([f for f,_ in window], win_label))
    return clips
