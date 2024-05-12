import torch
import torch.nn.functional as F
input = torch.randn(4,256,68,168)
kernel = torch.randn(256,1,11,11)
out = F.conv2d(input,kernel,padding=11//2,groups=256)
a=1