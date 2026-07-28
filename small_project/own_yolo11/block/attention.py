# import library
import torch
import torch.nn as nn

from ..layer import Conv

class Attention(nn.Module):
    
    # initialized
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        attn_ratio =0.5
    ):
        
        # nn.Module reset to use PyTorch
        super(Attention, self).__init__()
        
        # parameter
        ch_i = in_channels      # input channels
        ch_o = out_channels     # output channels
        head_cnt = num_heads    # attention head counter
        
        # save parameter
        self.out_channels = ch_o
        self.num_heads = head_cnt
        
        # channels per attetion head
        self.head_dim = ch_o // head_cnt
        
        # Query and Key channels per attention head
        self.qk_dim = int(
            self.head_dim * attn_ratio
        )
        
        # attention score scale
        self.scale = self.qk_dim ** -0.5
        
        # total Query or Key channels
        qk_channels = (
            self.qk_dim
            * head_cnt
        )
        
        # QKV Conv output channels
        qkv_channels = (
            qk_channels
            + qk_channels
            + ch_o
        )
        
        # QKV Conv
        # input: [B, ch, H, W]
        # output: [B, qkv_channels, H, W]
        self.qkv = Conv(
            in_channels=ch_i,
            out_channels=qkv_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation="identity"
        )
        
        # Position Encoding
        # Depthwise Conv:
        # groups = channels
        self.pe = Conv(
            in_channels=ch_o,
            out_channels=ch_o,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=ch_o,
            activation="identity"
        )
        
        # Projection Conv
        self.proj = Conv(
            in_channels=ch_o,
            out_channels=ch_o,
            kernel_size=1,
            stride=1,
            padding=0,
            activation="identity"
        )
        
    # forward
    def forward(self, x):
        
        # tensor size
        # B: batch size
        # C: channels
        # H: height
        # W: width
        B, _, H, W = x.shape
        
        # output channels
        C = self.out_channels
        
        # spatial position counter
        N = H * W
        
        # QKV Conv
        x_qkv = self.qkv(x)
        
        # Reshape
        # [B, qkv_channels, H, W]
        # ->
        # [
        #     B,
        #     num_heads,
        #     key_dim + key_dim + head_dim,
        #     N
        # ]
        x_qkv = x_qkv.view(
            B,
            self.num_heads,
            self.qk_dim * 2 + self.head_dim,
            N
        )
        
        # Split Query, Key and Value
        #
        # q: [B, num_heads, key_dim, N]
        # k: [B, num_heads, key_dim, N]
        # v: [B, num_heads, head_dim, N]
        q, k, v = x_qkv.split(
            [
                self.qk_dim,
                self.qk_dim,
                self.head_dim
            ],
            dim=2
        )
        
        # Attention score
        # q:
        # [B, num_heads, key_dim, N]
        # q transpose:
        # [B, num_heads, N, key_dim]
        # k:
        # [B, num_heads, key_dim, N]
        # attention:
        # [B, num_heads, N, N]
        attention = (
            q * self.scale
        ).transpose(-2, -1) @ k
        
        # Softmax
        attention = attention.softmax(
            dim=-1
        )
        
        # Apply attention weights to Value
        #
        # output:
        # [B, num_heads, head_dim, N]
        x = (
            v
            @ attention.transpose(-2, -1)
        )
        
        # Combine attention heads
        #
        # [B, num_heads, head_dim, N]
        #
        # ->
        #
        # [B, C, H, W]
        x = x.reshape(
            B,
            C,
            H,
            W
        )
        
        # Restore Value feature map
        value_feature = v.reshape(
            B,
            C,
            H,
            W
        )
        
        # Position Encoding
        position_encoding = self.pe(
            value_feature
        )
        
        # Attention output
        # + Position Encoding
        x = x + position_encoding
        
        
        # Projection Conv
        x = self.proj(x)
        
        # return
        return x