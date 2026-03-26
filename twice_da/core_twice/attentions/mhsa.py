import torch
from torch import nn, einsum
from timm.models.layers import trunc_normal_
from einops import rearrange
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 attn_dropout=0.,
                 proj_dropout=0.,
                 bias=False,
                 kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = float(head_dim) ** -0.5

        kdim = kdim or dim
        vdim = vdim or dim

        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, kdim, bias=bias)
        self.v = nn.Linear(dim, vdim, bias=bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, query, key, value):
        # Linear Projection of Q, K, V
        q, k, v = self.q(query), self.k(key), self.v(value)

        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        # Scaled dot-product attention
        attn = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
