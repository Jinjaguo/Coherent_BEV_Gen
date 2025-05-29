
import torch

tensor = torch.load("./bev_tensors/bev_0000.pt", map_location="cpu")  # 路径按需改一下
print("shape :", tensor.shape)
print("dtype :", tensor.dtype)
print("ndim  :", tensor.ndim)
