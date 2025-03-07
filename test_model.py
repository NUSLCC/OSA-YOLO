import torch
import torch.nn as nn
from ultralytics.nn.modules.block import A2C2f, HybridMambaAA
from ultralytics.nn.modules.mamba_yolo import VSSBlock

if __name__ == '__main__':
    
    x = torch.randn(1, 4, 40, 40) # batch_size, channels, height, width
    x = x.cuda()
    print(x.shape)
    
    model1 = A2C2f(c1=4, c2=64).cuda()
    y = model1(x)
    print(y.shape)

    model2 = VSSBlock(in_channels=4, hidden_dim=128).cuda()
    y = model2(x)
    print(y.shape)

    model3 = HybridMambaAA(c1=4, c2=256).cuda()
    y = model3(x)
    print(y.shape)