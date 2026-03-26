import torch
import torch.nn as nn
from einops import rearrange
from torchvision.ops import StochasticDepth
from timm.models import register_model
from core_twice.utils import Conv2D, EfficientChannelAttention
from core_twice.attentions.mhsa import MultiheadAttention
from core_twice.attentions.dmha import DeformableMultiheadAttention
from core_twice.utils import get_norm_layer

class OverlapPatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 patch_size: int,
                 overlap_size: int,
                 norm_type: str):
        super(OverlapPatchEmbedding, self).__init__()
        self.downsampling = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=overlap_size, padding=(patch_size // 2))
        self.norm = get_norm_layer(norm_type=norm_type, num_features=embedding_dim)

    def forward(self, x):
        x = self.norm(self.downsampling(x))
        return x

class FFN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 expansion_ratio: int,
                 mlp_dropout: float,
                 activation: torch.nn):
        super(FFN, self).__init__()
        hidden_features = in_channels * expansion_ratio
        self.linear_1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_features, kernel_size=1)
        self.activation = activation()
        self.linear_2 = nn.Conv2d(in_channels=hidden_features, out_channels=in_channels, kernel_size=1)
        self.dropout = nn.Dropout(mlp_dropout) if mlp_dropout > 0. else nn.Identity()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

class MultiScalePerceptionUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: list,
                 activation: torch.nn,
                 conv_dropout_rate: float):
        super(MultiScalePerceptionUnit, self).__init__()
        reducted_channels = out_channels
        self.pointwise_conv2d = nn.Conv2d(in_channels=in_channels, out_channels=reducted_channels, kernel_size=1)

        self.reduction_conv2d_list = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=reducted_channels, kernel_size=1) for i in range(len(kernel_sizes))])

        self.multiscale_conv2d_list = nn.ModuleList([Conv2D(in_channels=reducted_channels,
                                                            out_channels=reducted_channels,
                                                            kernel_size=kernel_sizes[i],
                                                            groups=reducted_channels,
                                                            activation=activation,
                                                            dropout_rate=conv_dropout_rate,
                                                            if_act=False,
                                                            if_batch_norm=False) for i in range(len(kernel_sizes))])

    def forward(self, x):
        branch_results = []
        for i in range(len(self.multiscale_conv2d_list)):
            projection = self.reduction_conv2d_list[i](x)
            branch_results.append(self.multiscale_conv2d_list[i](projection) + projection) # depth-wise + residual connection
        branch_results.append(self.pointwise_conv2d(x))
        branch_results = torch.cat(branch_results, dim=1)
        return branch_results

class StandardMHSA(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_sizes: list,
                 num_heads: int,
                 offset_groups: int,
                 kv_reduction_ratio: int,
                 activation: torch.nn,
                 norm_type: str,
                 conv_dropout_rate: float,
                 attention_dropout: float):
        super(StandardMHSA, self).__init__()
        reducted_channels = in_channels // (len(kernel_sizes) + 1)
        concat_channels = (reducted_channels) * (len(kernel_sizes) + 1)
        self.mspu = MultiScalePerceptionUnit(in_channels=in_channels,
                                             out_channels=reducted_channels,
                                             kernel_sizes=kernel_sizes,
                                             activation=activation,
                                             conv_dropout_rate=conv_dropout_rate)
        self.norm = get_norm_layer(norm_type=norm_type, num_features=concat_channels)
        self.eca = EfficientChannelAttention(kernel_size=9)
        self.mhsa = MultiheadAttention(dim=concat_channels,
                                       num_heads=num_heads,
                                       attn_dropout=attention_dropout,
                                       proj_dropout=0.0,
                                       bias=False)

        self.kv_compressor = nn.AvgPool2d(kernel_size=kv_reduction_ratio) # Pooling compression

    def forward(self, x):
        _, _, h, w = x.shape
        branch_results = self.mspu(x)
        branch_results = self.norm(branch_results)
        branch_results = self.eca(branch_results)
        compressed_features = rearrange(self.kv_compressor(branch_results), "b c h w -> b (h w) c")
        initial_features = rearrange(branch_results, "b c h w -> b (h w) c")
        mhsa_results = self.multi_head_self_attention(query=initial_features, key=compressed_features, value=compressed_features)
        new_features = rearrange(mhsa_results, "b (h w) c -> b c h w", h=h, w=w)
        return new_features

class DeformableMHSA(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_sizes: list,
                 num_heads: int,
                 offset_groups: int,
                 kv_reduction_ratio: int,
                 activation: torch.nn,
                 norm_type: str,
                 conv_dropout_rate: float,
                 attention_dropout: float):
        super(DeformableMHSA, self).__init__()
        reducted_channels = in_channels // (len(kernel_sizes) + 1)
        concat_channels = (reducted_channels) * (len(kernel_sizes) + 1)
        self.mspu = MultiScalePerceptionUnit(in_channels=in_channels,
                                             out_channels=reducted_channels,
                                             kernel_sizes=kernel_sizes,
                                             activation=activation,
                                             conv_dropout_rate=conv_dropout_rate)
        self.norm = get_norm_layer(norm_type=norm_type, num_features=concat_channels)
        self.eca = EfficientChannelAttention(kernel_size=9)
        self.dmha = DeformableMultiheadAttention(dim=concat_channels,
                                                 num_heads=num_heads,
                                                 offset_groups=offset_groups,
                                                 offset_scale=kv_reduction_ratio,
                                                 activation=activation,
                                                 norm_type='layer_norm',
                                                 attn_dropout=attention_dropout,
                                                 proj_dropout=0.0,
                                                 bias=False)

    def forward(self, x):
        x = self.mspu(x)
        x = self.norm(x)
        x = self.eca(x)
        x = self.dmha(x=x)
        return x

class TwiceBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_sizes: list,
                 num_heads: int,
                 offset_groups: int,
                 mlp_expansion_ratio: int,
                 kv_reduction_ratio: int,
                 activation: torch.nn,
                 norm_type: str,
                 conv_dropout_rate: float,
                 drop_path_rate: float,
                 mlp_dropout: float,
                 attention_dropout: float):
        super(TwiceBlock, self).__init__()
        self.norm_1 = get_norm_layer(norm_type=norm_type, num_features=in_channels)
        self.mhsa = DeformableMHSA(in_channels=in_channels,
                                   kernel_sizes=kernel_sizes,
                                   num_heads=num_heads,
                                   offset_groups=offset_groups,
                                   kv_reduction_ratio=kv_reduction_ratio,
                                   activation=activation,
                                   norm_type=norm_type,
                                   conv_dropout_rate=conv_dropout_rate,
                                   attention_dropout=attention_dropout)
        self.norm_2 = get_norm_layer(norm_type=norm_type, num_features=in_channels)
        self.dw_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, groups=in_channels, padding=1)
        self.norm_3 = get_norm_layer(norm_type=norm_type, num_features=in_channels)
        self.ffn = FFN(in_channels=in_channels, expansion_ratio=mlp_expansion_ratio, mlp_dropout=mlp_dropout, activation=activation)
        self.drop_path = StochasticDepth(drop_path_rate, mode="batch") if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        norm = self.norm_1(x)
        mhsa = self.mhsa(norm)
        mhsa = self.drop_path(mhsa)
        x = x + mhsa
        x = x + self.dw_conv(self.norm_2(x))
        ffn = self.norm_3(x)
        ffn = self.ffn(ffn)
        ffn = self.drop_path(ffn)
        x = x + ffn
        return x

class Stem(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: torch.nn,
                 norm_type: str):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=3, stride=2, padding=1),
                                  get_norm_layer(norm_type=norm_type, num_features=out_channels // 2),
                                  activation(),

                                  nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                                  get_norm_layer(norm_type=norm_type, num_features=out_channels),
                                  )

    def forward(self, x):
        return self.stem(x)

class TwiceEncoder(nn.Module):
    def __init__(self,
                 img_channels: int,
                 channels: list,
                 depth: list,
                 kernel_sizes: list,
                 num_heads: list,
                 offset_groups: list,
                 patch_sizes: list,
                 overlap_sizes: list,
                 mlp_expansion_ratios: list,
                 kv_reduction_ratios: list,
                 activation: torch.nn,
                 norm_type: str,
                 conv_dropout_rate: float,
                 drop_path_rate: float,
                 mlp_dropout: float,
                 attention_dropout: float):
        super(TwiceEncoder, self).__init__()
        self.stages = nn.ModuleList([])
        self.overlap_patch_embed = nn.ModuleList([])
        in_channels = [img_channels]
        in_channels.extend(channels)

        drop_path_values = self.prepare_dropout_values((0.0, drop_path_rate), depth, mode="ascending")

        for i in range(len(depth)):
            if(i == 0):
                self.overlap_patch_embed.append(Stem(in_channels=img_channels, out_channels=channels[i], activation=activation, norm_type=norm_type))
            else:
                self.overlap_patch_embed.append(OverlapPatchEmbedding(in_channels=in_channels[i],
                                                                      embedding_dim=channels[i],
                                                                      patch_size=patch_sizes[i],
                                                                      overlap_size=overlap_sizes[i],
                                                                      norm_type=norm_type))

            self.stages.append(nn.ModuleList([TwiceBlock(in_channels=channels[i],
                                                         kernel_sizes=kernel_sizes[i],
                                                         num_heads=num_heads[i],
                                                         offset_groups=offset_groups[i],
                                                         mlp_expansion_ratio=mlp_expansion_ratios[i],
                                                         kv_reduction_ratio=kv_reduction_ratios[i],
                                                         activation=activation,
                                                         norm_type=norm_type,
                                                         conv_dropout_rate=conv_dropout_rate,
                                                         drop_path_rate=drop_path_values[i][d],
                                                         mlp_dropout=mlp_dropout,
                                                         attention_dropout=attention_dropout
                                                         ) for d in range(depth[i])]))

    def forward(self, x):
        features_of_each_stage = []
        for s in range(len(self.stages)):
            x = self.overlap_patch_embed[s](x)
            for stage in self.stages[s]:
                x = stage(x)
            features_of_each_stage.append(x)
        return features_of_each_stage

    def prepare_dropout_values(self, min_max_value, depth, mode="descending"):
        """Создаёт список списков значений Dropout для всех уровней."""
        total_blocks = sum(depth)  # Общее количество блоков
        drop_values = []  # Список для всех уровней

        current_idx = 0
        for d in depth:
            level_values = [
                self.interpolate_drop_value(min_max_value, current_idx + idx, total_blocks, mode)
                for idx in range(d)
            ]
            drop_values.append(level_values)
            current_idx += d

        return drop_values

    def interpolate_drop_value(self, min_max_value, current_idx, total_idx, mode="descending"):
        min_value = min_max_value[0]
        max_value = min_max_value[1]
        # Вычисление интерполированного значения
        t = current_idx / (total_idx - 1)  # Нормализованный индекс
        if mode == "descending":
            drop_value = max_value - (max_value - min_value) * t
        elif mode == "ascending":
            drop_value = min_value + (max_value - min_value) * t
        return round(drop_value, 3)


class ClassificationHead(nn.Module):
    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 dropout: float
                 ):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.final_classifier = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.final_classifier(x)
        return x


class TwiceDA(nn.Module):
    def __init__(self,
                 img_channels: int,
                 channels: list,
                 depth: list,
                 kernel_sizes: list,
                 num_heads: list,
                 offset_groups: list,
                 patch_sizes: list,
                 overlap_sizes: list,
                 mlp_expansion_ratios: list,
                 kv_reduction_ratios: list,
                 activation: torch.nn,
                 norm_type: str,
                 conv_dropout_rate: float,
                 drop_path_rate: float,
                 mlp_dropout: float,
                 classificator_dropout: float,
                 attention_dropout: float,
                 num_classes: int):
        super(TwiceDA, self).__init__()
        self.encoder = TwiceEncoder(img_channels=img_channels,
                                    channels=channels,
                                    depth=depth,
                                    kernel_sizes=kernel_sizes,
                                    num_heads=num_heads,
                                    offset_groups=offset_groups,
                                    patch_sizes=patch_sizes,
                                    overlap_sizes=overlap_sizes,
                                    mlp_expansion_ratios=mlp_expansion_ratios,
                                    kv_reduction_ratios= kv_reduction_ratios,
                                    activation=activation,
                                    norm_type=norm_type,
                                    conv_dropout_rate=conv_dropout_rate,
                                    drop_path_rate=drop_path_rate,
                                    mlp_dropout=mlp_dropout,
                                    attention_dropout=attention_dropout)
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = get_norm_layer(norm_type=norm_type, num_features=channels[-1])
        self.flatten = nn.Flatten()
        self.classification_head = ClassificationHead(in_features=channels[-1], num_classes=num_classes, dropout=classificator_dropout)

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.global_pooling(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.classification_head(x)
        return x

@register_model
def twice_da_tiny(num_classes):
    model = TwiceDA(img_channels=3,
                    channels=[64, 128, 256, 512],
                    depth=[3, 3, 6, 3],
                    kernel_sizes=[[(3, 3), (7, 7), (21, 21)],
                                  [(3, 3), (7, 7), (15, 15)],
                                  [(3, 3), (5, 5), (11, 11)],
                                  [(3, 3), (5, 5), (7, 7)]],
                    num_heads=[2, 4, 8, 16],
                    offset_groups=[1, 2, 4, 8],
                    patch_sizes=[7, 3, 3, 3],
                    overlap_sizes=[4, 2, 2, 2],
                    mlp_expansion_ratios=[2, 2, 2, 2],
                    kv_reduction_ratios=[8, 4, 2, 1],
                    activation=nn.GELU,
                    norm_type='batch_norm',
                    conv_dropout_rate=0.0,
                    drop_path_rate=0.2,
                    mlp_dropout=0.2,
                    classificator_dropout=0.5,
                    attention_dropout=0.1,
                    num_classes=num_classes)
    return model