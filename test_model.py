import torch
import torch.nn as nn
from ultralytics.nn.modules.block import A2C2f, HybridMambaAA
from ultralytics.nn.modules.mamba_yolo import SS2D, SS2D_Zig, VSSBlock, VSSBlockOmni, XSSBlock, XSSBlockOmni

if __name__ == '__main__':
    
    x = torch.randn(1, 64, 40, 40).cuda() # batch_size, channels, height, width
    print(x.shape)
    
    # model1 = A2C2f(c1=64, c2=64).cuda()
    # y = model1(x)
    # print(y.shape)

    # model2 = VSSBlock(in_channels=64, hidden_dim=128).cuda()
    # y = model2(x)
    # print(y.shape)

    # model3 = HybridMambaAA(c1=64, c2=256).cuda()
    # y = model3(x)
    # print(y.shape)

    # model4 = VSSBlockOmni(in_channels=64, hidden_dim=128).cuda()
    # y = model4(x)
    # print(y.shape)

    # model5 = XSSBlockOmni(in_channels=64, hidden_dim=128).cuda()
    # y = model5(x)
    # print(y.shape)

    model = SS2D_Zig(d_model=64, d_state=128).cuda()
    y = model(x)
    print(y.shape)
    model = VSSBlock(in_channels=64, hidden_dim=128).cuda()
    y = model(x)
    print(y.shape)
    model = XSSBlock(in_channels=64, hidden_dim=128).cuda()
    y = model(x)
    print(y.shape)