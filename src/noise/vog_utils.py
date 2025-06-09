# src/noise/vog_utils.py
import torch


class GradientVarianceTracker:
    def __init__(self, feature_dim, num_samples):
        # 使用在线Welford算法记录每个样本的梯度均值和平方和
        self.feature_dim = feature_dim
        self.count = torch.zeros(num_samples)  # 每个样本已累积梯度次数（epoch数）
        self.mean_grad = torch.zeros(num_samples, feature_dim)  # 梯度均值
        self.M2 = torch.zeros(num_samples, feature_dim)  # 用于计算方差的累积变量

    def update(self, sample_indices, grad_vectors):
        """
        更新给定样本索引集合的梯度统计。
        sample_indices: 样本索引列表
        grad_vectors: 张量 [n, feature_dim]，对应这些样本的梯度向量
        """
        for idx, grad in zip(sample_indices, grad_vectors):
            idx = int(idx)  # 转为Python整数索引
            self.count[idx] += 1
            # 更新均值和M2（Welford在线算法）
            delta = grad - self.mean_grad[idx]
            self.mean_grad[idx] += delta / self.count[idx]
            delta2 = grad - self.mean_grad[idx]
            self.M2[idx] += delta * delta2

    def get_variances(self):
        """
        返回每个样本的梯度方差向量的范数，作为难度分数。
        采用每个样本梯度向量方差的L2范数作为标量分数。
        """
        # 避免除零，对 count < 2 的样本方差设为0（数据不足计算方差）
        var = torch.zeros_like(self.count)
        for i in range(len(self.count)):
            if self.count[i] > 1:
                # 方差 = M2/(n-1)
                var_vector = self.M2[i] / (self.count[i] - 1)
                # 使用方差向量的2范数作为难度得分
                var[i] = torch.norm(var_vector, p=2)
            else:
                var[i] = 0.0
        return var
