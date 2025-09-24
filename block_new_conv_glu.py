from conv import ConvMultiScaleFeatureExtractor
from vss import VSSBlock
import torch.nn as nn
from sban import SBAN
from conv_glu import ConvolutionalGLU


class ConvVSSNetwork(nn.Module):
    def __init__(self,
                 hidden_dim,
                 drop_path=0.1,
                 attn_drop=0.,
                 norm_layer=nn.LayerNorm,
                 d_state: int = 16):
        super(ConvVSSNetwork, self).__init__()

        # 全局特征提取 (VSS)
        self.global_feature_extractor = VSSBlock(
            hidden_dim=hidden_dim // 2,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop,
            d_state=d_state,
        )
        # 局部特征提取
        # self.local_feature_extractor = ConvMultiScaleFeatureExtractor(
        #     in_channels=hidden_dim // 2, out_channels=hidden_dim // 2)

        # 局部特征提取
        self.local_feature_extractor = ConvolutionalGLU(in_features=hidden_dim//2)

        # 添加 SBAN 模块
        self.sban = SBAN(inc=[hidden_dim // 2, hidden_dim // 2], input_dim=hidden_dim)

    def forward(self, input):
        # 通道分半
        input_top, input_bottom = input.chunk(2, dim=-1)

        # 全局特征提取 (VSS)
        global_feature = self.global_feature_extractor(input_bottom)  # (batch_size, width, height, channel_size)

        # 局部特征提取
        local_feature = input_top.permute(0, 3, 1, 2).contiguous()  # (batch_size, channel_size, width, height)
        local_feature = self.local_feature_extractor(local_feature)  # (batch_size, channel_size, width, height)
        local_feature = local_feature.permute(0, 2, 3, 1).contiguous()  # (batch_size, width, height, channel_size)

        # 将 global_feature 和 local_feature 转换为 SBAN 需要的形状
        global_feature_for_sban = global_feature.permute(0, 3, 1, 2).contiguous()  # (batch_size, channel_size, width, height)
        local_feature_for_sban = local_feature.permute(0, 3, 1, 2).contiguous()  # (batch_size, channel_size, width, height)

        # 使用 SBAN 进行特征融合
        fused_feature = self.sban((global_feature_for_sban, local_feature_for_sban))  # (batch_size, channel_size, width, height)

        # 将 fused_feature 转换回原始形状
        output = fused_feature.permute(0, 2, 3, 1).contiguous()  # (batch_size, width, height, channel_size)

        return input + output