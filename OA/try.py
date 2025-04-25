from torchvision import models
import torch
import torch
import torch.nn as nn
from torchvision import models



# checkpoint = torch.hub.load_state_dict_from_url(
#     'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth', map_location='cpu', check_hash=True)
# print(checkpoint['model'].keys())
# print(checkpoint['model']['pos_embed'].shape[-1])

model = torch.load('/root/autodl-tmp/capstone/Backbone_3/data/model')
print(model.keys())


import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())


import torch
from torch import nn
import torchvision
from torchvision import models
from timm.models import create_model
from .evit import *



def get_cnn_model(args):

    model = models.vgg19(pretrained = args.pretrained)

    num_ftrs = model.classifier[6].in_features
    feature_model = list(model.classifier.children())
    feature_model.pop()
    feature_model.append(nn.Linear(num_ftrs, args.num_class))
    model.classifier = nn.Sequential(*feature_model)

    return model

class CNN_TFV_CrossAttention(nn.Module):
    def __init__(self, args, default_cfg=None):
        super(CNN_TFV_CrossAttention, self).__init__()

        self.single_v = args.single_v
        self.visit_num = args.visit_num
        self.num_class = args.num_class
        self.base_keep_rate = args.base_keep_rate

        v00_backbone = get_cnn_model(args=args)
        v12_backbone = get_cnn_model(args=args)
        v24_backbone = get_cnn_model(args=args)

        self.v00_backbone_256 = v00_backbone.features[0:19]
        self.v00_backbone_512 = v00_backbone.features[19:37]
        self.v00_avg = v00_backbone.avgpool
        self.v00_cls = v00_backbone.classifier

        self.v12_backbone_256 = v12_backbone.features[0:19]
        self.v12_backbone_512 = v12_backbone.features[19:37]
        self.v12_avg = v12_backbone.avgpool
        self.v12_cls = v12_backbone.classifier

        self.v24_backbone_256 = v24_backbone.features[0:19]
        self.v24_backbone_512 = v24_backbone.features[19:37]
        self.v24_avg = v24_backbone.avgpool
        self.v24_cls = v24_backbone.classifier

        self.base_keep_rate = args.base_keep_rate

        self.v00_256_FTV = create_model(
                'deit_small_patch2_28_shrink',
                pretrained=False,
                num_classes=args.num_class,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                fuse_token=args.fuse_token,
                img_size=(28, 28),
                in_chans=256
            )
        self.v12_256_FTV = create_model(
                'deit_small_patch2_28_shrink',
                pretrained=False,
                num_classes=args.num_class,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                fuse_token=args.fuse_token,
                img_size=(28, 28),
                in_chans=256
            )
        self.v24_256_FTV = create_model(
                'deit_small_patch2_28_shrink',
                pretrained=False,
                num_classes=args.num_class,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                fuse_token=args.fuse_token,
                img_size=(28, 28),
                in_chans=256
            )
        self.fusion_linear = nn.Linear(args.feature_dim*3, args.num_class)


    def forward(self, v00, v12, v24):
        v00 = self.v00_backbone_256(v00)
        v12 = self.v12_backbone_256(v12)
        v24 = self.v24_backbone_256(v24)

        # 判断是否为单点测试模式
        if self.single_v:
            if self.visit_num == 'v00':
                v = v00
                out, _ = self.v00_256_FTV(v, self.base_keep_rate, get_idx=True)
            elif self.visit_num == 'v12':
                v = v12
                out, _ = self.v12_256_FTV(v, self.base_keep_rate, get_idx=True)
            elif self.visit_num == 'v24':
                v = v24
                out, _ = self.v24_256_FTV(v, self.base_keep_rate, get_idx=True)
            else:
                raise ValueError(f"Invalid visit_num: {self.visit_num}")
            return out

        # 三帧处理（多视图融合）
        v00_result, v00_256_idx = self.v00_256_FTV(v00, self.base_keep_rate, get_idx=True)
        v12_result, v12_256_idx = self.v12_256_FTV(v12, self.base_keep_rate, get_idx=True)
        v24_result, v24_256_idx = self.v24_256_FTV(v24, self.base_keep_rate, get_idx=True)

        # 融合并输出
        fusion = torch.cat((v00_result, v12_result, v24_result), dim=1)   # concat or add or multiply or attention
        output = (v00_result + v12_result + v24_result) / 3
        return output



#######################################################################
import torch
from torch import nn
import torchvision
from torchvision import models
from timm.models import create_model
from .evit import *



def get_cnn_model(args):

    model = models.vgg19(pretrained = args.pretrained)

    num_ftrs = model.classifier[6].in_features
    feature_model = list(model.classifier.children())
    feature_model.pop()
    feature_model.append(nn.Linear(num_ftrs, args.num_class))
    model.classifier = nn.Sequential(*feature_model)

    return model

class CNN_TFV_CrossAttention(nn.Module):
    def __init__(self, args):
        super(CNN_TFV_CrossAttention, self).__init__()

        v00_backbone = get_cnn_model(args=args)
        v12_backbone = get_cnn_model(args=args)
        v24_backbone = get_cnn_model(args=args)

        self.v00_backbone_256 = v00_backbone.features[0:19]
        self.v00_backbone_512 = v00_backbone.features[19:37]
        self.v00_avg = v00_backbone.avgpool
        self.v00_cls = v00_backbone.classifier

        self.v12_backbone_256 = v12_backbone.features[0:19]
        self.v12_backbone_512 = v12_backbone.features[19:37]
        self.v12_avg = v12_backbone.avgpool
        self.v12_cls = v12_backbone.classifier

        self.v24_backbone_256 = v24_backbone.features[0:19]
        self.v24_backbone_512 = v24_backbone.features[19:37]
        self.v24_avg = v24_backbone.avgpool
        self.v24_cls = v24_backbone.classifier

        self.base_keep_rate = args.base_keep_rate

        self.v00_256_FTV = create_model(
                'deit_small_patch2_28_shrink',
                pretrained=False,
                num_classes=args.num_class,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                fuse_token=args.fuse_token,
                img_size=(28, 28),
                in_chans=256
            )
        self.v12_256_FTV = create_model(
                'deit_small_patch2_28_shrink',
                pretrained=False,
                num_classes=args.num_class,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                fuse_token=args.fuse_token,
                img_size=(28, 28),
                in_chans=256
            )
        self.v24_256_FTV = create_model(
                'deit_small_patch2_28_shrink',
                pretrained=False,
                num_classes=args.num_class,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                fuse_token=args.fuse_token,
                img_size=(28, 28),
                in_chans=256
            )


    def forward(self, v00, v12, v24):
        v00 = self.v00_backbone_256(v00)
        v12 = self.v12_backbone_256(v12)
        v24 = self.v24_backbone_256(v24)

        v00_result, v00_256_idx = self.v00_256_FTV(v00, self.base_keep_rate, get_idx=True)
        v12_result, v12_256_idx = self.v12_256_FTV(v12, self.base_keep_rate, get_idx=True)
        v24_result, v24_256_idx = self.v24_256_FTV(v24, self.base_keep_rate, get_idx=True)

        return v00_result, v12_result, v24_result
