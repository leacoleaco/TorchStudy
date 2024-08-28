import torch
from torch import nn

src = torch.rand((4, 1, 2))
tgt = torch.rand((1, 1, 2))
# >> > out = transformer_model(src, tgt)

transformer_model = nn.Transformer(d_model=2, nhead=8, num_encoder_layers=12)

# src = torch.tensor([
#     [1],
#     [2],
#     [3],
#     [4],
# ], dtype=torch.float32)
#
# # 目标序列
# tgt = torch.tensor([[5],
#                     [6]], dtype=torch.float32)

out = transformer_model(src, tgt)

print(out.shape)
