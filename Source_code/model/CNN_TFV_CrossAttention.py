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
