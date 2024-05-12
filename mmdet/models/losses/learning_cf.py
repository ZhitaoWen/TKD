import torch
a = torch.tensor([[1,2,3],
                [4,5,6]])
b = torch.tensor([[4,5,6],[7,8,9]])
print(a*b,torch.mul(a,b))