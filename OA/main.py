import os, sys
import argparse
import torch
from dataset.loader import data_load
from model.CNN_model import CNN_Fusion
from model.ViT_3_fusion import ViT
from model.VGG_ViT import VGG_ViT_Exchange, VGG_ViT_Fusion
from train_eng import train_model
from eval_eng import visualise_TFV
import model.Fusion_ViT
from timm.models import create_model
from model.CNN_TFV_CrossAttention import CNN_TFV_CrossAttention
from model.test_gateattention_512 import CNN_TFV_Gateattention
# from model.cnn_vgg19_gateattention_256 import CNN_TFV_SelfGateAttention
from model.single_test import CNN_TFV_CrossAttentiongate
from model.single_test512 import CNN_TFV_Gate512
from model.CNN_ATTENTION import CNN_Gate_Multiview
from model.VIT_GATEATTENTION import ViT_GateAttention
from model.CNN_LSTM import CNN_LSTM


def set_args():
    parser = argparse.ArgumentParser(description='Pytorch Multi-Slice MRI Knee OA Severity Grading Training')
    # Initial Para
    parser.add_argument('--device',                         type=int,   default=0)
    parser.add_argument('--best_model_name',                type=str,   default="try__l")
    # Data Para
    parser.add_argument('--data_dir',                       type=str,   default="/root/autodl-tmp/v3_data")
    parser.add_argument('--model_dir',                      type=str,   default="/root/autodl-tmp/data/model")
    parser.add_argument('--batch_size',                     type=int,   default=32)
    # Model Para
    parser.add_argument('--model_type',                     type=str,   default='cnn_lstm') # use_vit; vgg_vit_exchange; vgg_vit_fusion; cnn; vit_token_fusion; cnn_tfv_crossattention; for_v00_gateattention; for_CNN_TFV_CrossAttentiongate; CNN_TFV_Gate512; CNN_Gate_Multiview; ViT_GateAttention; cnn_lstm
    parser.add_argument('--num_class',                      type=int,   default=5)
    parser.add_argument('--pretrained',                     type=bool,  default=True)
    parser.add_argument('--debug',                          type=bool,  default=False)
    # Para for CNN-based model
    parser.add_argument('--net_type',                       type=str,   default='resnet') # resnet; vgg; densenet; inception
    parser.add_argument('--depth',                          type=str,   default='50') # 18, 34, 50, 101, 152; 16, 19, 16bn, 19bn; 121, 169, 201; v3
    parser.add_argument('--feature_dim',                    type=int,   default=256)
    # Para for using only one visit in CNN-based model
    parser.add_argument('--single_v',                       type=bool,  default=False)
    parser.add_argument('--visit_num',                      type=str,   default='v00')
    # Para for VGG-ViT fusion model
    parser.add_argument('--fusion_type',                    type=str,   default='concat') # concat; multiply; add; mean; attention  ##concat好使 ##add好使(12个epoch之后)（--塌缩） ##mean也好使 ##multiply（不好使，跟add效果一样，会塌缩）##attention(也不好使)
    # Para for ViT Token Fusion Model
    parser.add_argument('--vit_fusion_model_name',          type=str,   default='deit_base_patch16_224', metavar='MODEL')# deit_base_patch16_224 deit_small_patch16_224_shrink(这个其实可以注释掉，用不到)
    parser.add_argument('--drop',                           type=float, default=0.0, metavar='PCT')
    parser.add_argument('--drop-path',                      type=float, default=0.1, metavar='PCT')
    parser.add_argument('--fuse_token',                     action='store_true')
    parser.add_argument('--base_keep_rate',                 type=float, default=0.7)
    parser.add_argument('--drop_loc',                       type=str,   default='(3, 6, 9)')
    parser.add_argument('--finetune',                       type=str,   default="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth")# https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
    parser.add_argument('--visual_TFV',                     type=str,   default="")
    # Para for CNN + LSTM
    parser.add_argument('--lstm_hidden_dim',                type=int, default=128, help='Hidden dimension for LSTM layer')
    parser.add_argument('--lstm_num_layers',                type=int, default=1, help='Number of LSTM layers (default: 1)')

    # Training Para
    parser.add_argument('--num_epoch',                      type=int,   default=100)
    parser.add_argument('--optim',                          type=str,   default='SGD') # 'AdamW' 'Adam' 'RMSprop' 'SGD'  ##'AdamW'(在单点上的效果太差,时序上会在第22个epoch开始上升) ##'Adam'也是 ##'SGD'比较可以，能够学习出东西 ##
    parser.add_argument('--lr',                             type=float, default=5.0e-3)
    parser.add_argument('--lr_decay_epoch',                 type=int,   default=10)
    parser.add_argument('--weight_decay',                   type=float, default=5e-4)
    # Para for retraining and testing
    parser.add_argument('--load_model',                     type=bool,  default=False)
    parser.add_argument('--load_model_dir',                 type=str,   default="")

    args = parser.parse_args()
    return args

def checkpoint(model, args):
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
        return model


if __name__ == '__main__':

    args = set_args()
    if args.device == 0:
        device = torch.device('cuda:0')
    elif args.device == 1:
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')

    print('--Phase 1: Data prepration--')
    dset_loaders, dset_size, num_class = data_load(args)
    args.num_class = num_class

    if args.model_type != 'vit_token_fusion':
        args.visual_TFV = ''

    if args.visual_TFV:
        print('--Phase 2: Model setup--')
        model = torch.load(args.visual_TFV)
        model.to(device=device)
        print('--Phase 3: Visualise Model--')
        visualise_TFV(args, model, dset_loaders)
    else:
        print('--Phase 2: Model setup--')
        if args.single_v:
            args.feature_dim = args.num_class

        if args.model_type == 'use_vit':
            model = ViT(image_size=224, patch_size=56, num_classes=args.num_class, dim=128, depth=12, heads=8, mlp_dim=768)
        elif args.model_type == 'vgg_vit_exchange':
            model = VGG_ViT_Exchange(args)
        elif args.model_type == 'vgg_vit_fusion':
            model = VGG_ViT_Fusion(args)
        elif args.model_type == 'cnn':
            model = CNN_Fusion(args)
        elif args.model_type == 'cnn_tfv_crossattention':
            model = CNN_TFV_CrossAttention(args)
        elif args.model_type == 'for_v00_gateattention':
            model = CNN_TFV_Gateattention(args)
        elif args.model_type == 'for_CNN_TFV_CrossAttentiongate':
            model = CNN_TFV_CrossAttentiongate(args)
        elif args.model_type == 'CNN_TFV_Gate512':
            model = CNN_TFV_Gate512(args)
        elif args.model_type == 'CNN_Gate_Multiview':
            model = CNN_Gate_Multiview(args)
        elif args.model_type == 'ViT_GateAttention':
            model = ViT_GateAttention(args)
        elif args.model_type == 'cnn_lstm':
            model = CNN_LSTM(args)
        elif args.model_type == 'vit_token_fusion':
            # model = create_model(
            #     args.vit_fusion_model_name,
            #     keep_rate=(args.base_keep_rate, ),
            #     # drop_loc=eval(args.drop_loc),
            #     pretrained=False,
            #     num_classes=args.num_class,
            #     drop_rate=args.drop,
            #     drop_path_rate=args.drop_path,
            #     drop_block_rate=None,
            #     fuse_token=args.fuse_token,
            #     img_size=(224, 224)
            # )
            model = create_model(
                args.vit_fusion_model_name,
                # keep_rate=(args.base_keep_rate, ),
                # drop_loc=eval(args.drop_loc),
                pretrained=False,
                num_classes=args.num_class,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                fuse_token=args.fuse_token,
                img_size=(224, 224)
            )
            model = checkpoint(model, args)
        model.to(device=device)

        print('--Phase 3: Model training--')
        train_model(args, model, dset_loaders, dset_size)

