import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
from model.grid_attention_layer import GridAttentionBlock2D_TORR

class ViT_GateAttention(nn.Module):
    def __init__(self, args):
        super(ViT_GateAttention, self).__init__()

        self.single_v = args.single_v
        self.visit_num = args.visit_num
        self.num_class = args.num_class
        self.base_keep_rate = args.base_keep_rate
        self.fusion_type = args.fusion_type

        # === ViT 模型 ===
        def build_vit():
            vit = create_model(
                "vit_base_patch16_224",
                pretrained=True,
                num_classes=0,
                img_size=(224, 224),
                in_chans=3,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
            )
            vit.reset_classifier(num_classes=args.num_class)  # 重置为自己的类别数
            return vit

        self.v00_vit = build_vit()
        self.v12_vit = build_vit()
        self.v24_vit = build_vit()

        # === Gate Attention 用于 transformer 输出 token ===
        self.att_v00 = GridAttentionBlock2D_TORR(768, 768, 256)
        self.att_v12 = GridAttentionBlock2D_TORR(768, 768, 256)
        self.att_v24 = GridAttentionBlock2D_TORR(768, 768, 256)

        self.concat_linear = nn.Linear(args.num_class * 3, args.num_class)

    def vit_feat_to_grid(self, x):
        """
        兼容两种返回格式：
        1. [B, N, C] → ViT Token 输出（有 CLS）
        2. [B, C] → 已经 pooled 的输出（不适合做 attention）
        """
        if isinstance(x, tuple):
            x = x[0]

        if x.dim() == 2:
            raise ValueError(
                "forward_features(Error) ")

        x = x[:, 1:, :]
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        return x.permute(0, 2, 1).reshape(B, C, H, W)  # [B, C, H, W]



    def forward(self, v00=None, v12=None, v24=None):
        if self.single_v:
            if self.visit_num == 'v00':
                feat = self.vit_feat_to_grid(self.v00_vit.forward_features(v00))
                gated, _ = self.att_v00(feat, feat)
                pooled = F.adaptive_avg_pool2d(gated, 1).view(gated.size(0), -1)
                return self.v00_vit.head(pooled)
            elif self.visit_num == 'v12':
                feat = self.vit_feat_to_grid(self.v12_vit.forward_features(v12))
                gated, _ = self.att_v12(feat, feat)
                pooled = F.adaptive_avg_pool2d(gated, 1).view(gated.size(0), -1)
                return self.v12_vit.head(pooled)
            elif self.visit_num == 'v24':
                feat = self.vit_feat_to_grid(self.v24_vit.forward_features(v24))
                gated, _ = self.att_v24(feat, feat)
                pooled = F.adaptive_avg_pool2d(gated, 1).view(gated.size(0), -1)
                return self.v24_vit.head(pooled)

        # 多帧融合路径
        f00 = self.vit_feat_to_grid(self.v00_vit.forward_features(v00))
        f12 = self.vit_feat_to_grid(self.v12_vit.forward_features(v12))
        f24 = self.vit_feat_to_grid(self.v24_vit.forward_features(v24))

        g00, _ = self.att_v00(f00, f00)
        g12, _ = self.att_v12(f12, f12)
        g24, _ = self.att_v24(f24, f24)

        pooled00 = F.adaptive_avg_pool2d(g00, 1).view(g00.size(0), -1)
        pooled12 = F.adaptive_avg_pool2d(g12, 1).view(g12.size(0), -1)
        pooled24 = F.adaptive_avg_pool2d(g24, 1).view(g24.size(0), -1)

        fusion_feat = torch.cat([pooled00, pooled12, pooled24], dim=1)
        return self.concat_linear(fusion_feat)
