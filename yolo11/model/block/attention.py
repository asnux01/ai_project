#----------------------------------------------
# 라이브러리
#----------------------------------------------
import torch.nn as nn

from ..layer import Conv

class Attention(nn.Module):
    
    # 초기화
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        attn_ratio =0.5
    ):
        
        # PyTorch 사용을 위해 nn.Module 초기화
        super(Attention, self).__init__()
        
        # 파라미터
        ch_i = in_channels      # 입력 채널
        ch_o = out_channels     # 출력 채널
        head_cnt = num_heads    # attention head 카운터
        
        # 포워드 접근 가능 파라미터 
        # 출력 파라미터
        self.out_channels = ch_o
        
        # attention head 카운터
        self.num_heads = head_cnt
        
        # 각 attention head의 출력 채널
        self.head_dim = ch_o // head_cnt
        
        # Query와 Key 채널 수
        self.qk_dim = int(
            self.head_dim * attn_ratio
        )
        
        # Attention score(Query와 Key 내적 크기 억제)
        self.scale = self.qk_dim ** -0.5
        
        # 모든 head의 Query 또는 Key 채널 수
        qk_channels = (
            self.qk_dim
            * head_cnt
        )
        
        # QKV Conv 출력 채널 수
        qkv_channels = (
            qk_channels
            + qk_channels
            + ch_o
        )
        
        # QKV Conv
        # 입력: [B, ch, H, W]
        # 출력: [B, qkv_channels, H, W]
        # projection 역할이므로 활성화는 사용 x
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
        
        # Attention 결과를 최종 Conv로 projection
        self.proj = Conv(
            in_channels=ch_o,
            out_channels=ch_o,
            kernel_size=1,
            stride=1,
            padding=0,
            activation="identity"
        )
        
    # 포워드
    def forward(self, x):
        
        # 입력 텐서: [B, C, H, W]
        # B: 배치(Batch) 크기
        # C: 채널 수
        # H: 높이
        # W: 너비
        B, _, H, W = x.shape
        
        # output channels
        C = self.out_channels
        
        # 모든 공간 위치를 하나의 sequence 길이로 취급
        N = H * W
        
        # QKV Conv
        x_qkv = self.qkv(x)
        
        # 텐서 형태 변환
        # [B, qkv_channels, H, W]
        # ->
        # [B, num_heads, 2 * qk_dim + head_dim, N]
        x_qkv = x_qkv.view(
            B,
            self.num_heads,
            self.qk_dim * 2 + self.head_dim,
            N
        )
        
        # Query, Key, Value 분리
        # q: [B, num_heads, qk_dim, N]
        # k: [B, num_heads, qk_dim, N]
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
        # [B, num_heads, qk_dim, N]
        # q transpose:
        # [B, num_heads, N, qk_dim]
        # k:
        # [B, num_heads, qk_dim, N]
        # attention:
        # [B, num_heads, N, N]
        attention = (
            q * self.scale
        ).transpose(-2, -1) @ k
        
        # Softmax
        attention = attention.softmax(
            dim=-1
        )
        
        # attention weights를 Value에 가중
        #
        # 결과:
        # [B, num_heads, head_dim, N]
        x = (
            v
            @ attention.transpose(-2, -1)
        )
        
        # attention head들을 결합
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
        
        # Value도 원래 형태로 복원
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
        
        # 반환
        return x