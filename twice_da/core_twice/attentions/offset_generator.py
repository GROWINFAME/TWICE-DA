import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from core_twice.utils import get_norm_layer

class MultiScaleOffsetGenerator2D(nn.Module):
    def __init__(self,
                 offset_dim,
                 offset_kernel_sizes,
                 offset_scale,
                 activation,
                 norm_type):
        super(MultiScaleOffsetGenerator2D, self).__init__()
        self.offset_scale = offset_scale
        self.projection_conv = nn.ModuleList([nn.Conv2d(in_channels=offset_dim // 4, out_channels=1, kernel_size=1) for i in range(4)])
        self.multiscale_depthwise_conv = nn.ModuleList([nn.Conv2d(in_channels=12,
                                                                  out_channels=12,
                                                                  kernel_size=offset_kernel_sizes[i],
                                                                  groups=12,
                                                                  stride=offset_scale,
                                                                  padding=(offset_kernel_sizes[i] - 1) // 2)
                                                        for i in range(len(offset_kernel_sizes))
                                                        ])
        self.layer_norm = get_norm_layer(norm_type=norm_type, num_features=12 * len(offset_kernel_sizes))
        self.activation = activation()
        self.pointwise_conv = nn.Conv2d(in_channels=12 * len(offset_kernel_sizes),
                                        out_channels=2,
                                        kernel_size=1,
                                        bias=False)
        self.tanh_act = nn.Tanh()

        nn.init.zeros_(self.pointwise_conv.weight)

    def apply_offsets(self, x, offsets):
        N, _, H_in, W_in = x.shape
        _, _, H_out, W_out = offsets.shape

        # === 1. Строим базовую сетку в пиксельных координатах входа ===
        # yy, xx = матрицы координат (0..H_out-1, 0..W_out-1)
        ys = torch.arange(H_out, device=x.device, dtype=x.dtype)
        xs = torch.arange(W_out, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (H_out, W_out)

        # Переводим координаты из low-res (H_out, W_out) в пиксели входа (H_in, W_in)
        # Один шаг по low-res соответствует (H_in/H_out, W_in/W_out) пикселей на входе
        gx = xx * (W_in / W_out)  # координата X в пикселях входа
        gy = yy * (H_in / H_out)  # координата Y в пикселях входа

        # === 2. Нормализация в диапазон [-1, 1] ===
        # формула для align_corners=False: (coord + 0.5) / size * 2 - 1
        gx = (gx + 0.5) / W_in * 2 - 1  # (H_out, W_out)
        gy = (gy + 0.5) / H_in * 2 - 1  # (H_out, W_out)

        # Базовая сетка (x, y) в нормализованных координатах
        base_grid = torch.stack((gx, gy), dim=-1)  # (H_out, W_out, 2)
        base_grid = base_grid.unsqueeze(0).expand(N, -1, -1, -1)  # (N, H_out, W_out, 2)

        # === 3. Добавляем offsets ===
        # offsets тоже должны быть в нормализованных координатах (в пределах [-1,1]),
        # чтобы шаги были сопоставимы с base_grid.
        # Например, offset = 0.1 означает смещение на 10% размера по соответствующей оси.
        offsets = offsets.permute(0, 2, 3, 1)  # (N, H_out, W_out, 2)

        # Финальная сетка для grid_sample
        grid = base_grid + offsets
        grid = grid.clamp(-1, 1)  # чтобы не улетать за границы

        # === 4. Выборка значений ===
        out = F.grid_sample(
            x, grid,
            mode='bilinear',
            padding_mode='border',  # более стабильно, чем 'zeros'
            align_corners=False
        )
        return out

    def forward(self, x):
        # Разделение тензора на 4 части по оси каналов
        parts = torch.chunk(x, chunks=4, dim=1)

        # Применение операций к каждой части
        processed_parts = [
            torch.cat([
                self.agg_channel(part, "max"),
                self.agg_channel(part, "avg"),
                self.projection_conv[i](part)
            ], dim=1) for i, part in enumerate(parts)
        ]

        # Объединение всех обработанных частей
        x = torch.cat(processed_parts, dim=1)
        # Применение depthwise свёрток
        x = torch.cat([conv(x) for conv in self.multiscale_depthwise_conv], dim=1)
        x = self.layer_norm(x)
        x = self.activation(x)
        # Применение pointwise свёртки
        x = self.pointwise_conv(x)
        x = self.tanh_act(x)
        return x

    def agg_channel(self, x, pool = "max"):
        b,c,h,w = x.size()
        x = rearrange(x, "b c h w -> b (h w) c")
        x = F.max_pool1d(x, c) if pool == "max" else F.avg_pool1d(x, c)
        x = rearrange(x, "b (h w) 1 -> b 1 h w", h=h, w=w)
        return x

class OffsetGenerator2D(nn.Module):
    def __init__(self,
                 offset_dim,
                 offset_kernel_sizes,
                 offset_scale):
        super(OffsetGenerator2D, self).__init__()
        self.offset_scale = offset_scale
        self.to_offsets = nn.Sequential(
            nn.Conv2d(offset_dim, offset_dim, offset_kernel_sizes[0], groups=offset_dim, stride=offset_scale, padding=(offset_kernel_sizes[0] - offset_scale) // 2),
            nn.GELU(),
            nn.Conv2d(offset_dim, 2, 1, bias = False),
            nn.Tanh(),
        )

    def apply_offsets(self, x, offsets):
        N, _, H_in, W_in = x.shape
        _, _, H_out, W_out = offsets.shape

        # === 1. Строим базовую сетку в пиксельных координатах входа ===
        # yy, xx = матрицы координат (0..H_out-1, 0..W_out-1)
        ys = torch.arange(H_out, device=x.device, dtype=x.dtype)
        xs = torch.arange(W_out, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (H_out, W_out)

        # Переводим координаты из low-res (H_out, W_out) в пиксели входа (H_in, W_in)
        # Один шаг по low-res соответствует (H_in/H_out, W_in/W_out) пикселей на входе
        gx = xx * (W_in / W_out)  # координата X в пикселях входа
        gy = yy * (H_in / H_out)  # координата Y в пикселях входа

        # === 2. Нормализация в диапазон [-1, 1] ===
        # формула для align_corners=False: (coord + 0.5) / size * 2 - 1
        gx = (gx + 0.5) / W_in * 2 - 1  # (H_out, W_out)
        gy = (gy + 0.5) / H_in * 2 - 1  # (H_out, W_out)

        # Базовая сетка (x, y) в нормализованных координатах
        base_grid = torch.stack((gx, gy), dim=-1)  # (H_out, W_out, 2)
        base_grid = base_grid.unsqueeze(0).expand(N, -1, -1, -1)  # (N, H_out, W_out, 2)

        # === 3. Добавляем offsets ===
        # offsets тоже должны быть в нормализованных координатах (в пределах [-1,1]),
        # чтобы шаги были сопоставимы с base_grid.
        # Например, offset = 0.1 означает смещение на 10% размера по соответствующей оси.
        offsets = offsets.permute(0, 2, 3, 1)  # (N, H_out, W_out, 2)

        # Финальная сетка для grid_sample
        grid = base_grid + offsets
        grid = grid.clamp(-1, 1)  # чтобы не улетать за границы

        # === 4. Выборка значений ===
        out = F.grid_sample(
            x, grid,
            mode='bilinear',
            padding_mode='border',  # более стабильно, чем 'zeros'
            align_corners=False
        )
        return out

    def forward(self, x):
        x = self.to_offsets(x)
        return x