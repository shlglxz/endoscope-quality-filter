# src/active/active_sampler.py
import torch
import math


class ActiveSampler:
    def __init__(self, method="entropy"):
        self.method = method

    def select(self, model, dataset, candidate_indices, k, device=None):
        """
        从候选索引列表中根据策略选择 k 个样本索引进行标注。
        model: 当前模型
        dataset: 训练数据集 (ClipDataset)
        candidate_indices: 未清洗样本索引列表
        k: 要选择的样本数量
        """
        model.eval()
        if self.method == "entropy":
            # 熵采样: 计算每个候选样本的预测熵，选择熵最大的k个
            entropies = []
            for idx in candidate_indices:
                clip_tensor, noisy_label, orig_label = dataset[idx]
                clip_tensor = clip_tensor.unsqueeze(0)
                if device:
                    clip_tensor = clip_tensor.to(device)
                logits, _ = model(clip_tensor)
                probs = torch.softmax(logits.cpu(), dim=1).squeeze(0)  # 模型预测概率
                # 计算信息熵 -sum(p * log p)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                entropies.append((entropy, idx))
            # 按熵值降序排序，选择前k个
            entropies.sort(reverse=True, key=lambda x: x[0])
            selected = [idx for (_, idx) in entropies[:k]]

        elif self.method == "core-set":
            # 核心集采样: 使用贪心远点算法在特征空间选择覆盖最大差异的k个样本
            # 计算所有候选样本的clip特征向量
            features = []
            for idx in candidate_indices:
                clip_tensor, _, _ = dataset[idx]
                clip_tensor = clip_tensor.unsqueeze(0)
                if device:
                    clip_tensor = clip_tensor.to(device)
                _, feat = model(clip_tensor)  # 提取clip特征
                features.append(feat.cpu().detach().numpy().reshape(-1))
            features = np.array(features)  # shape [N_candidates, feature_dim]
            selected = []
            if len(features) == 0:
                return selected
            # 贪心选择初始样本
            selected.append(candidate_indices[0])
            # 特征空间计算距离并迭代选择
            selected_feats = [features[0]]
            unselected = list(range(1, len(candidate_indices)))
            while len(selected) < k and unselected:
                # 计算每个未选样本到已选集合的最近距离
                max_dist = -1
                max_idx = -1
                for j in unselected:
                    # 计算此样本与当前所有已选样本的最小距离
                    dists = [np.linalg.norm(features[j] - sf) for sf in selected_feats]
                    dist_to_set = min(dists)
                    if dist_to_set > max_dist:
                        max_dist = dist_to_set
                        max_idx = j
                # 选择距离最远的样本
                selected.append(candidate_indices[max_idx])
                selected_feats.append(features[max_idx])
                unselected.remove(max_idx)
        else:
            raise ValueError(f"Unknown active sampling method: {self.method}")

        return selected
