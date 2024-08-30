import torch

a = torch.Tensor([[[1, 2, 3]], [[4, 5, 6]]])
print(a)
print(a.shape)

b = a.unsqueeze(1)
print(b)
print(b.shape)

c = a.squeeze(1)
print(c)
print(c.shape)
