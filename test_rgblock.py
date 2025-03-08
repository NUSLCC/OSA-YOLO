import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import CBAM


class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) Module.
    
    It computes channel-wise attention weights via global average pooling followed by a 
    1D convolution and a sigmoid activation, without any dimensionality reduction.
    
    Args:
        channel (int): Number of input channels.
        k_size (int): Kernel size for the 1D convolution. Default: 3.
    """
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D convolution with padding to maintain the channel dimension
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)               # shape: [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2) # shape: [B, 1, C]
        y = self.conv(y)                   # shape: [B, 1, C]
        y = self.sigmoid(y)                # shape: [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # shape: [B, C, 1, 1]
        return x * y.expand_as(x)


class RGBlockECA(nn.Module):
    """
    RGBlock with Efficient Channel Attention.
    
    This block uses a bottleneck design to process the input features:
      1. Expand channels via a 1x1 convolution.
      2. Apply a depthwise convolution for spatial feature extraction.
      3. Project back to the original number of channels.
      4. Add a residual connection.
      5. Use ECA to reweight the output channels.
      
    Args:
        in_features (int): Number of input (and output) channels.
        expansion (int): Channel expansion factor. Default is 2.
        drop (float): Dropout rate applied after the projection. Default is 0.
        k_size (int): Kernel size for the ECA module. Default is 3.
    """
    def __init__(self, in_features, expansion=2, drop=0.0, k_size=3):
        super().__init__()
        hidden_features = in_features * expansion
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.bn = nn.BatchNorm2d(hidden_features)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_features, in_features, kernel_size=1)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.eca = ECA(in_features, k_size=k_size)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.dwconv(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.drop(out)
        out = out + residual  # Residual connection
        out = self.eca(out)     # Apply efficient channel attention
        return out


class RCBAMBlock(nn.Module):
    """
    An RG Block that first applies a bottleneck transformation with a residual connection,
    then refines the output with CBAM.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = in_features * 2
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, 
                                groups=hidden_features)
        self.bn = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(hidden_features, in_features, kernel_size=1)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.cbam = CBAM(in_features)
        self.conv3 = nn.Conv2d(in_features, out_features, kernel_size=1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.dwconv(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.drop(out)
        out = out + x
        out = self.cbam(out)
        out = self.conv3(out)
        return out
    

# Example usage:
if __name__ == "__main__":
    dummy_input = torch.randn(1, 64, 32, 32)  # [B, C, H, W]

    block_eca = RGBlockECA(in_features=64, expansion=2, drop=0.1, k_size=3)
    output = block_eca(dummy_input)
    print("Output ECA shape:", output.shape)

    block_cbam = RCBAMBlock(in_features=64, out_features=32, drop=0.1)    
    output = block_cbam(dummy_input)
    print("Output CBAM shape:", output.shape)    
