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

def load_pretrained_vit(model, finetune_path):
    if finetune_path.startswith("http"):
        checkpoint = torch.hub.load_state_dict_from_url(finetune_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(finetune_path, map_location='cpu')

    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
        
    incompatible_keys = []
    state_dict = model.state_dict()
    for k in list(checkpoint.keys()):
        if k in state_dict and checkpoint[k].shape != state_dict[k].shape:
            incompatible_keys.append(k)
            del checkpoint[k]

    msg = model.load_state_dict(checkpoint, strict=False)
    print("Loaded pretrained ViT with message:", msg)
    if incompatible_keys:
        print("Skipped keys due to shape mismatch:", incompatible_keys)

class CNN_TFV_Gate512(nn.Module):
    def __init__(self, args, default_cfg=None):
        super(CNN_TFV_Gate512, self).__init__()

        self.single_v = args.single_v
        self.visit_num = args.visit_num
        self.num_class = args.num_class
        self.base_keep_rate = args.base_keep_rate
        self.fusion_type = args.fusion_type

        # === Backbones for v00, v12, v24 ===
        self.v00_backbone = get_cnn_model(args=args)
        self.v12_backbone = get_cnn_model(args=args)
        self.v24_backbone = get_cnn_model(args=args)

        # Extract 256 and 512 feature blocks
        self.v00_feat_256 = self.v00_backbone.features[:19]  # conv1_1 ~ conv4_4
        self.v00_feat_512 = self.v00_backbone.features[19:37]  # conv5_x

        self.v12_feat_256 = self.v12_backbone.features[:19]
        self.v12_feat_512 = self.v12_backbone.features[19:37]

        self.v24_feat_256 = self.v24_backbone.features[:19]
        self.v24_feat_512 = self.v24_backbone.features[19:37]

        # === Attention blocks ===
        self.att_v00 = GridAttentionBlock2D_TORR(256, 512, 128, mode='concatenation_sigmoid')
        self.att_v12 = GridAttentionBlock2D_TORR(256, 512, 128, mode='concatenation_sigmoid')
        self.att_v24 = GridAttentionBlock2D_TORR(256, 512, 128, mode='concatenation_sigmoid')

        # === Vision Transformers ===
        self.v00_vit = create_model('deit_base_patch16_224', pretrained=False, num_classes=args.num_class,
                                    drop_rate=args.drop, drop_path_rate=args.drop_path, fuse_token=args.fuse_token,
                                    img_size=(28, 28), in_chans=256)
        load_pretrained_vit(self.v00_vit, args.finetune)

        self.v12_vit = create_model('deit_base_patch16_224', pretrained=False, num_classes=args.num_class,
                                    drop_rate=args.drop, drop_path_rate=args.drop_path, fuse_token=args.fuse_token,
                                    img_size=(28, 28), in_chans=256)
        load_pretrained_vit(self.v12_vit, args.finetune)

        self.v24_vit = create_model('deit_base_patch16_224', pretrained=False, num_classes=args.num_class,
                                    drop_rate=args.drop, drop_path_rate=args.drop_path, fuse_token=args.fuse_token,
                                    img_size=(28, 28), in_chans=256)
        load_pretrained_vit(self.v24_vit, args.finetune)

        self.concat_linear = nn.Linear(args.num_class * 3, args.num_class)

    def forward(self, v00=None, v12=None, v24=None):
        if self.single_v:
            if self.visit_num == 'v00':
                x256 = self.v00_feat_256(v00)
                x512 = self.v00_feat_512(x256)
                x512 = F.interpolate(x512, size=x256.shape[2:], mode='bilinear', align_corners=False)
                att, att_map = self.att_v00(x256, x512)
                self.att_map_v00 = att_map
                out, _ = self.v00_vit(att, self.base_keep_rate, get_idx=True)
                return out
            elif self.visit_num == 'v12':
                x256 = self.v12_feat_256(v12)
                x512 = self.v12_feat_512(x256)
                x512 = F.interpolate(x512, size=x256.shape[2:], mode='bilinear', align_corners=False)
                att, _ = self.att_v12(x256, x512)
                out, _ = self.v12_vit(att, self.base_keep_rate, get_idx=True)
                return out
            elif self.visit_num == 'v24':
                x256 = self.v24_feat_256(v24)
                x512 = self.v24_feat_512(x256)
                x512 = F.interpolate(x512, size=x256.shape[2:], mode='bilinear', align_corners=False)
                att, _ = self.att_v24(x256, x512)
                out, _ = self.v24_vit(att, self.base_keep_rate, get_idx=True)
                return out

        # === Multi-view fusion ===
        x256_00 = self.v00_feat_256(v00)
        x512_00 = self.v00_feat_512(x256_00)
        x512_00 = F.interpolate(x512_00, size=x256_00.shape[2:], mode='bilinear', align_corners=False)
        att00, _ = self.att_v00(x256_00, x512_00)
        out00, _ = self.v00_vit(att00, self.base_keep_rate, get_idx=True)

        x256_12 = self.v12_feat_256(v12)
        x512_12 = self.v12_feat_512(x256_12)
        x512_12 = F.interpolate(x512_12, size=x256_12.shape[2:], mode='bilinear', align_corners=False)
        att12, _ = self.att_v12(x256_12, x512_12)
        out12, _ = self.v12_vit(att12, self.base_keep_rate, get_idx=True)

        x256_24 = self.v24_feat_256(v24)
        x512_24 = self.v24_feat_512(x256_24)
        x512_24 = F.interpolate(x512_24, size=x256_24.shape[2:], mode='bilinear', align_corners=False)
        att24, _ = self.att_v24(x256_24, x512_24)
        out24, _ = self.v24_vit(att24, self.base_keep_rate, get_idx=True)

        fused = torch.cat([out00, out12, out24], dim=1)
        out = self.concat_linear(fused)
        return out
