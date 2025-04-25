import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from timm.models import create_model
from model.grid_attention_layer import GridAttentionBlock2D_TORR


def get_cnn_model(args):
    model = models.vgg19(pretrained=args.pretrained)
    num_ftrs = model.classifier[6].in_features
    feature_model = list(model.classifier.children())
    feature_model.pop()
    feature_model.append(nn.Linear(num_ftrs, args.num_class))
    model.classifier = nn.Sequential(*feature_model)
    return model


class CNN_TFV_CrossAttentiongate(nn.Module):
    def __init__(self, args, default_cfg=None):
        super(CNN_TFV_CrossAttentiongate, self).__init__()

        self.single_v = args.single_v
        self.visit_num = args.visit_num
        self.num_class = args.num_class
        self.base_keep_rate = args.base_keep_rate
        self.fusion_type = args.fusion_type

        # === VGG Backbone for each visit ===
        v00_backbone = get_cnn_model(args=args)
        v12_backbone = get_cnn_model(args=args)
        v24_backbone = get_cnn_model(args=args)

        self.v00_backbone_256 = v00_backbone.features[0:19]
        self.v12_backbone_256 = v12_backbone.features[0:19]
        self.v24_backbone_256 = v24_backbone.features[0:19]

        # === Grid Attention Block (Self-guided on 256) ===
        self.att_block_v00 = GridAttentionBlock2D_TORR(
            in_channels=256, gating_channels=256, inter_channels=128,
            mode='concatenation_sigmoid', sub_sample_factor=(1, 1), use_W=True
        )

        self.att_block_v12 = GridAttentionBlock2D_TORR(
            in_channels=256, gating_channels=256, inter_channels=128,
            mode='concatenation_sigmoid', sub_sample_factor=(1, 1), use_W=True
        )

        self.att_block_v24 = GridAttentionBlock2D_TORR(
            in_channels=256, gating_channels=256, inter_channels=128,
            mode='concatenation_sigmoid', sub_sample_factor=(1, 1), use_W=True
        )

        # === ViT ===
        self.v00_256_FTV = create_model(
            'deit_base_patch16_224', pretrained=False, num_classes=args.num_class,
            drop_rate=args.drop, drop_path_rate=args.drop_path, fuse_token=args.fuse_token,
            img_size=(28, 28), in_chans=256)

        self.v12_256_FTV = create_model(
            'deit_base_patch16_224', pretrained=False, num_classes=args.num_class,
            drop_rate=args.drop, drop_path_rate=args.drop_path, fuse_token=args.fuse_token,
            img_size=(28, 28), in_chans=256)

        self.v24_256_FTV = create_model(
            'deit_base_patch16_224', pretrained=False, num_classes=args.num_class,
            drop_rate=args.drop, drop_path_rate=args.drop_path, fuse_token=args.fuse_token,
            img_size=(28, 28), in_chans=256)
        self.concat_linear = nn.Linear(self.num_class * 3, self.num_class)

    def forward(self, v00=None, v12=None, v24=None):
        # === Feature extraction ===
        if self.single_v:
            if self.visit_num == 'v00':
                assert v00 is not None, "v00 input is None in single_v mode"
                v00_feat = self.v00_backbone_256(v00)
                v00_att, _ = self.att_block_v00(v00_feat, v00_feat)
                out, _ = self.v00_256_FTV(v00_att, self.base_keep_rate, get_idx=True)
                return out

            elif self.visit_num == 'v12':
                assert v12 is not None, "v12 input is None in single_v mode"
                v12_feat = self.v12_backbone_256(v12)
                v12_att, _ = self.att_block_v12(v12_feat, v12_feat)
                out, _ = self.v12_256_FTV(v12_att, self.base_keep_rate, get_idx=True)
                return out

            elif self.visit_num == 'v24':
                assert v24 is not None, "v24 input is None in single_v mode"
                v24_feat = self.v24_backbone_256(v24)
                v24_att, _ = self.att_block_v24(v24_feat, v24_feat)
                out, _ = self.v24_256_FTV(v24_att, self.base_keep_rate, get_idx=True)
                return out

            else:
                raise ValueError(f"Invalid visit_num: {self.visit_num}")
            return out


        # === 多点序列模式 ===
        assert all(x is not None for x in [v00, v12, v24]), "Multi-input mode requires all inputs"
        # === Feature extraction for all visits ===
        v00_feat = self.v00_backbone_256(v00)
        v12_feat = self.v12_backbone_256(v12)
        v24_feat = self.v24_backbone_256(v24)

        # === Self-guided attention ===
        v00_att, _ = self.att_block_v00(v00_feat, v00_feat)
        v12_att, _ = self.att_block_v12(v12_feat, v12_feat)
        v24_att, _ = self.att_block_v24(v24_feat, v24_feat)

        v00_out, _ = self.v00_256_FTV(v00_att, self.base_keep_rate, get_idx=True)
        v12_out, _ = self.v12_256_FTV(v12_att, self.base_keep_rate, get_idx=True)
        v24_out, _ = self.v24_256_FTV(v24_att, self.base_keep_rate, get_idx=True)

        fused = torch.cat([v00_out, v12_out, v24_out], dim=1)
        out = self.concat_linear(fused)
        return out
