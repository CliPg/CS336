import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

# 在 dim=1 上求和，不保留维度
sum1 = x.sum(dim=1, keepdim=False)
# 结果 shape: [2] （原来的第二个维度消失了）
print(sum1, sum1.shape)

# 在 dim=1 上求和，保留维度
sum2 = x.sum(dim=1, keepdim=True)
# 结果 shape: [2, 1] （第二个维度还在，只是长度为 1）
print(sum2, sum2.shape)
