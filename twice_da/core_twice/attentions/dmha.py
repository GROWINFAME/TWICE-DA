import torch
from torch import nn, einsum
from timm.models.layers import trunc_normal_
from einops import rearrange
import numpy as np
from core_twice.attentions.offset_generator import MultiScaleOffsetGenerator2D

class DeformableMultiheadAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 offset_groups,
                 offset_scale,
                 activation,
                 norm_type,
                 attn_dropout=0.,
                 proj_dropout=0.,
                 bias=False,
                 kdim=None,
                 vdim=None):
        super(DeformableMultiheadAttention, self).__init__()
        self.embed_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        #self.scale = float(head_dim) ** -0.5 # масштабирующий коэффициент 1 / sqrt(head_dim)
        self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1) / np.sqrt(head_dim)) # обучаемый масштабирующий коэффициент

        self.offset_groups = offset_groups
        offset_dim = dim // offset_groups

        kdim = kdim or dim
        vdim = vdim or dim

        self.offset_generator = MultiScaleOffsetGenerator2D(offset_dim=offset_dim, offset_kernel_sizes=self.generate_offset_kernel_sizes(offset_scale), offset_scale=offset_scale, activation=activation, norm_type=norm_type)

        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, kdim, bias=bias)
        self.v = nn.Linear(dim, vdim, bias=bias)
        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout > 0. else nn.Identity()
        self.drop_key = DropKey(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout) if proj_dropout > 0. else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def generate_offset_kernel_sizes(self, stride):
        if stride >= 8:
            kernel_sizes = [9, 15]
        elif stride >= 4:
            kernel_sizes = [5, 11]
        elif stride >= 2:
            kernel_sizes = [3, 7]
        else:
            kernel_sizes = [3, 5]
        return kernel_sizes

    def forward(self, x):
        # Rearrange input to 1D
        _, _, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")

        # Linear Projection of Q
        q = self.q(x)

        # Offset generation and getting only important features
        group = lambda t: rearrange(t, "b (h w) (g c) -> (b g) c h w", h=h, w=w, g=self.offset_groups) # разделяем каналы на g групп и обрабатываем параллельно одновременно каждую группу отдельно, это позволяет каждой группе генерировать различные смещения и, соответственно, выбирать разные key и value для механизма внимания.
        offsets = self.offset_generator(group(q))
        kv_features = self.offset_generator.apply_offsets(x=group(x), offsets=offsets)
        kv_features = rearrange(kv_features, '(b g) c h w -> b (h w) (g c)', b=x.size(0))

        # Linear Projection of K, V
        k, v = self.k(kv_features), self.v(kv_features)

        # Split into heads. Reshape to (batch_size, num_heads, seq_len, head_dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        # Scaled dot-product attention
        attn = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        #attn = self.drop_key(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)

        # Rearrange output to 2D
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out

class DropKey(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.mask_ratio = dropout_rate

    def forward(self, x):
        if self.training:
            mask = torch.ones_like(x) * self.mask_ratio  # вероятность маскирования
            drop_mask = torch.bernoulli(mask)  # 1 - замаскировать, 0 - сохранить
            return x + drop_mask * -1e12  # замаскированные элементы получают -1e12
        else:
            return x