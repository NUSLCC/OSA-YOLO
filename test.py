import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionalAttention(nn.Module):
    def __init__(self, in_channels, attn_hidden_channels, num_directions=8):
        """
        Computes an attention map for each of the 8 directional feature maps.
        
        Args:
            in_channels (int): Number of input channels per directional feature map.
            attn_hidden_channels (int): Hidden channels for the attention subnetwork.
            num_directions (int): Number of directional feature maps (default is 8).
        """
        super().__init__()
        self.num_directions = num_directions
        # Create one attention submodule per direction.
        self.attn_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, attn_hidden_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(attn_hidden_channels, 1, kernel_size=1)
            )
            for _ in range(num_directions)
        ])
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 8, D, H, W) where each slice along
                              the second dimension corresponds to a directional feature map.
                              
        Returns:
            torch.Tensor: Attention tensor of shape (B, 8, 1, H, W)
                          that can be used to modulate the directional features.
        """
        B, K, D, H, W = x.shape
        assert K == self.num_directions, f"Expected {self.num_directions} directional feature maps but got {K}."
        
        attn_outs = []
        for k in range(self.num_directions):
            # Extract the k-th directional feature map: shape (B, D, H, W)
            feat = x[:, k]
            # Compute raw attention scores for this direction: shape (B, 1, H, W)
            attn_k = self.attn_modules[k](feat)
            # Flatten spatial dimensions (H, W) and apply softmax so that weights sum to 1 per map
            attn_k = attn_k.view(B, -1)  # shape (B, H*W)
            attn_k = F.softmax(attn_k, dim=1)
            attn_k = attn_k.view(B, 1, H, W)  # reshape back to (B, 1, H, W)
            attn_outs.append(attn_k)
            
        # Stack the attention maps from all directions: shape (B, 8, 1, H, W)
        attn = torch.stack(attn_outs, dim=1)
        return attn

# Example usage:
if __name__ == "__main__":
    B, D, H, W = 2, 32, 16, 16  # Batch, channel per direction, height, width
    # Simulated input with 8 directional feature maps.
    ys = torch.randn(B, 8, D, H, W)
    
    # Create the directional attention module.
    directional_attn = DirectionalAttention(in_channels=D, attn_hidden_channels=16, num_directions=8)
    
    # Compute the attention weights.
    attn = directional_attn(ys)
    print("Attention weights shape:", attn.shape)  # Expected: (B, 8, 1, H, W)
    
    # These attention weights can now be applied in your CrossMerge_Omni_Attn function:
    # ys_modulated = ys * attn
