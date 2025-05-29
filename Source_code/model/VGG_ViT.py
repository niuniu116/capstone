import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .Transformer import TransformerModel
from .ViT_3_fusion import ViT, pair, Transformer

def get_cnn_model(args):

    model = models.vgg19(pretrained = args.pretrained)

    num_ftrs = model.classifier[6].in_features
    feature_model = list(model.classifier.children())
    feature_model.pop()
    feature_model.append(nn.Linear(num_ftrs, args.num_class))
    model.classifier = nn.Sequential(*feature_model)

    return model

class exchange_model(nn.Module):
    def __init__(self, exchange_n_embd):
        super(exchange_model, self).__init__()

        self.n_embd = exchange_n_embd
        self.exchange_dropout_rate = 0.1
        self.exchange_attn_dropout_rate = 0.1
        self.embd_pdrop = 0.1
        self.exchange_avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.pos_emb = nn.Parameter(torch.zeros(1, 3 * 7 * 7, exchange_n_embd))
        self.drop = nn.Dropout(self.embd_pdrop)

        self.exchange_transformer = TransformerModel(dim=self.n_embd, depth=2, heads=4, mlp_dim=2048,
                                                       dropout_rate=self.exchange_dropout_rate,
                                                       attn_dropout_rate=self.exchange_attn_dropout_rate)
        self.apply(self._init_weights)

    def forward(self, v00, v12, v24):
        batch_size = v00.size(0)
        if self.n_embd != 512:
            v00 = self.exchange_avgpool(v00)
            v12 = self.exchange_avgpool(v12)
            v24 = self.exchange_avgpool(v24)

        v00 = v00.view(batch_size, 1, -1, 7, 7)
        v12 = v12.view(batch_size, 1, -1, 7, 7)
        v24 = v24.view(batch_size, 1, -1, 7, 7)

        # v00 = repeat(v00, 'b () c h w -> b v c h w', v=1)
        # v12 = repeat(v12, 'b () c h w -> b v c h w', v=1)
        # v24 = repeat(v24, 'b () c h w -> b v c h w', v=1)

        token_embeddings = torch.cat((v00, v12, v24), dim=1).permute(0, 1, 3, 4, 2).contiguous()
        token_embeddings = token_embeddings.view(batch_size, -1, self.n_embd)

        pos_emb = self.pos_emb
        transformer_input = pos_emb + token_embeddings
        x = self.drop(transformer_input)
        x, _ = self.exchange_transformer(x)
        x = x.view(batch_size, 3, 7, 7, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3).contiguous()

        v00 = x[:, 0, :, :, :].contiguous().view(batch_size, -1, 7, 7)
        v12 = x[:, 1, :, :, :].contiguous().view(batch_size, -1, 7, 7)
        v24 = x[:, 2, :, :, :].contiguous().view(batch_size, -1, 7, 7)


        return v00, v12, v24


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class VGG_ViT_Exchange(nn.Module):
    def __init__(self, args):
        super(VGG_ViT_Exchange, self).__init__()

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

        self.transformer_exchange_256 = exchange_model(exchange_n_embd=256)
        self.transformer_exchange_512 = exchange_model(exchange_n_embd=512)

        self.fusion_linear = nn.Linear(15, args.num_class)


    def forward(self, v00, v12, v24):
        v00 = self.v00_backbone_256(v00)
        v12 = self.v12_backbone_256(v12)
        v24 = self.v24_backbone_256(v24)
        v00_256, v12_256, v24_256 = self.transformer_exchange_256(v00, v12, v24)
        v00_256 = F.interpolate(v00_256, scale_factor=4, mode='bilinear')
        v12_256 = F.interpolate(v12_256, scale_factor=4, mode='bilinear')
        v24_256 = F.interpolate(v24_256, scale_factor=4, mode='bilinear')
        v00 = v00 + v00_256
        v12 = v12 + v12_256
        v24 = v24 + v24_256

        v00 = self.v00_backbone_512(v00)
        v12 = self.v12_backbone_512(v12)
        v24 = self.v24_backbone_512(v24)
        v00_512, v12_512, v24_512 = self.transformer_exchange_512(v00, v12, v24)
        v00 = v00 + v00_512
        v12 = v12 + v12_512
        v24 = v24 + v24_512

        v00 = self.v00_avg(v00)
        v12 = self.v12_avg(v12)
        v24 = self.v24_avg(v24)
        v00 = torch.flatten(v00, 1)
        v12 = torch.flatten(v12, 1)
        v24 = torch.flatten(v24, 1)
        v00 = self.v00_cls(v00)
        v12 = self.v12_cls(v12)
        v24 = self.v24_cls(v24)

        fusion = torch.concat((v00, v12, v24), dim=1)
        result = self.fusion_linear(fusion)

        return result

class VGG_ViT_Fusion(nn.Module):
    def __init__(self, args):
        super(VGG_ViT_Fusion, self).__init__()

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

        self.fusion_type = args.fusion_type
        assert self.fusion_type in {'concat', 'add', 'multiply'}, 'fusion type must be either concat, add or multiply'
        if self.fusion_type == 'concat':
            self.transformer_exchange_256 = ViT(image_size=28, patch_size=7, num_classes=5, dim=4096, depth=12, heads=8, mlp_dim=1024, channels=256)
            self.transformer_exchange_512 = ViT(image_size=7, patch_size=1, num_classes=5, dim=128, depth=12, heads=8, mlp_dim=128, channels=512)
        elif self.fusion_type == 'add' or self.fusion_type == 'multiply':
            self.transformer_exchange_256 = None
            self.transformer_exchange_512 = None # thinking why need add or multiply in this case, what the meaning of the operation

        self.fusion_linear = nn.Linear(10, args.num_class)


    def forward(self, v00, v12, v24):
        v00 = self.v00_backbone_256(v00)
        v12 = self.v12_backbone_256(v12)
        v24 = self.v24_backbone_256(v24)

        if self.fusion_type == 'concat':
            f_256 = self.transformer_exchange_256(v00, v12, v24)
        elif self.fusion_type == 'add' or self.fusion_type == 'multiply':# thinking why need add or multiply in this case, what the meaning of the operation
            tran = torch.mul((v00, v12, v24))
            fusion_f = torch.mul((v00, v12, v24))
            f_256 = self.transformer_exchange_256(fusion_f)


        v00 = self.v00_backbone_512(v00)
        v12 = self.v12_backbone_512(v12)
        v24 = self.v24_backbone_512(v24)
        f_512 = self.transformer_exchange_512(v00, v12, v24)

        # v00 = self.v00_avg(v00)
        # v12 = self.v12_avg(v12)
        # v24 = self.v24_avg(v24)
        # v00 = torch.flatten(v00, 1)
        # v12 = torch.flatten(v12, 1)
        # v24 = torch.flatten(v24, 1)
        # v00 = self.v00_cls(v00)
        # v12 = self.v12_cls(v12)
        # v24 = self.v24_cls(v24)
        # fusion = torch.concat((v00, v12, v24), dim=1)

        fusion = torch.concat((f_256, f_512), dim=1)

        result = self.fusion_linear(fusion)

        return result

class Single_ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.sing_visit_to_patch_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                                       p1 = patch_height, p2 = patch_width)

        self.to_patch_embedding = nn.Linear(patch_dim, dim)


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):

        x = self.sing_visit_to_patch_embedding(x)
        x = self.to_patch_embedding(x)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
