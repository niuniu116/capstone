import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.grid_attention_layer import GridAttentionBlock2D_TORR

def get_vgg_parts(args):
    model = models.vgg19(pretrained=args.pretrained)
    return model.features[:19], model.features[19:37]

class CNN_Gate_Multiview(nn.Module):
    def __init__(self, args):
        super(CNN_Gate_Multiview, self).__init__()

        self.single_v = args.single_v
        self.visit_num = args.visit_num
        self.num_class = args.num_class

        # ========== 三个视图的 CNN + Gate ==========
        self.v00_feat_256, self.v00_feat_512 = get_vgg_parts(args)
        self.v12_feat_256, self.v12_feat_512 = get_vgg_parts(args)
        self.v24_feat_256, self.v24_feat_512 = get_vgg_parts(args)

        self.att_v00 = GridAttentionBlock2D_TORR(256, 512, 128)
        self.att_v12 = GridAttentionBlock2D_TORR(256, 512, 128)
        self.att_v24 = GridAttentionBlock2D_TORR(256, 512, 128)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_v00 = nn.Sequential(
            self.v00_feat_256,
            self.v00_feat_512,
        )
        self.cnn_v12 = nn.Sequential(
            self.v12_feat_256,
            self.v12_feat_512,
        )
        self.cnn_v24 = nn.Sequential(
            self.v24_feat_256,
            self.v24_feat_512,
        )

        # 最终线性层：三视图拼接
        self.fusion_linear = nn.Linear(256 * 3, self.num_class)
        self.cls_v00 = nn.Linear(256, self.num_class)
        self.cls_v12 = nn.Linear(256, self.num_class)
        self.cls_v24 = nn.Linear(256, self.num_class)

    def extract_feat(self, x256_layer, x512_layer, att_layer, x):
        x256 = x256_layer(x)
        x512 = x512_layer(x256)
        x512_up = F.interpolate(x512, size=x256.shape[2:], mode='bilinear', align_corners=False)
        att_feat, _ = att_layer(x256, x512_up)
        pooled = self.pool(att_feat)
        return pooled.view(pooled.size(0), -1)  # shape: (B, 256)

    def forward(self, v00=None, v12=None, v24=None):
        if self.single_v:
            if self.visit_num == 'v00':
                feat = self.extract_feat(self.v00_feat_256, self.v00_feat_512, self.att_v00, v00)
                return self.cls_v00(feat)
            elif self.visit_num == 'v12':
                feat = self.extract_feat(self.v12_feat_256, self.v12_feat_512, self.att_v12, v12)
                return self.cls_v12(feat)
            elif self.visit_num == 'v24':
                feat = self.extract_feat(self.v24_feat_256, self.v24_feat_512, self.att_v24, v24)
                return self.cls_v24(feat)
            else:
                raise ValueError("Invalid visit_num")

        # 多视角融合模式
        f00 = self.extract_feat(self.v00_feat_256, self.v00_feat_512, self.att_v00, v00)
        f12 = self.extract_feat(self.v12_feat_256, self.v12_feat_512, self.att_v12, v12)
        f24 = self.extract_feat(self.v24_feat_256, self.v24_feat_512, self.att_v24, v24)

        fusion_feat = torch.cat([f00, f12, f24], dim=1)  # (B, 256*3)
        return self.fusion_linear(fusion_feat)
