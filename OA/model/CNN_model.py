import torch
import torch.nn as nn
from torchvision import models

def get_cnn_model(args):

    if args.net_type == "resnet":
        if args.depth == "18":
            model = models.resnet18(pretrained = args.pretrained)
        elif args.depth == "34":
            model = models.resnet34(pretrained = args.pretrained)
        elif args.depth == "50":
            model = models.resnet50(pretrained = args.pretrained)
        elif args.depth == "101":
            model = models.resnet101(pretrained = args.pretrained)
        elif args.depth == "152":
            model = models.resnet152(pretrained = args.pretrained)
        else:
            return None

        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, args.feature_dim)

    elif args.net_type == "vgg":
        if args.depth == "16":
            model = models.vgg16(pretrained = args.pretrained)
        elif args.depth == "19":
            model = models.vgg19(pretrained = args.pretrained)
        elif args.depth == "16bn":
            model = models.vgg16_bn(pretrained = args.pretrained)
        elif args.depth == "19bn":
            model = models.vgg19_bn(pretrained = args.pretrained)
        else:
            return None

        num_ftrs = model.classifier[6].in_features
        feature_model = list(model.classifier.children())
        feature_model.pop()
        feature_model.append(nn.Linear(num_ftrs, args.feature_dim))
        model.classifier = nn.Sequential(*feature_model)

    elif args.net_type == "densenet":
        if args.depth == "121":
            model = models.densenet121(pretrained = args.pretrained)
        elif args.depth == "169":
            model = models.densenet169(pretrained = args.pretrained)
        elif args.depth == "201":
            model = models.densenet201(pretrained = args.pretrained)
        else:
            return None

        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, args.feature_dim)

    elif args.net_type == "inception":
        if args.depth == "v3":
            model = models.inception_v3(pretrained = args.pretrained)
        else:
            return None

        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, args.feature_dim)

    else:
        return None

    return model

class CNN_Fusion(nn.Module):
    def __init__(self, args):
        super(CNN_Fusion, self).__init__()

        self.single_v = args.single_v
        self.visit_num = args.visit_num

        self.cnn_v00 = get_cnn_model(args=args)
        self.cnn_v12 = get_cnn_model(args=args)
        self.cnn_v24 = get_cnn_model(args=args)

        self.fusion_linear = nn.Linear(args.feature_dim*3, args.num_class)

    def forward(self, v00, v12, v24):
        if self.single_v:
            if self.visit_num == 'v00':
                result = self.cnn_v00(v00)
                return result
            elif self.visit_num == 'v12':
                result = self.cnn_v12(v12)
                return result
            elif self.visit_num == 'v24':
                result = self.cnn_v24(v24)
                return result
            else:
                print('Error: invalid visit number')
                return None
        v00_f = self.cnn_v00(v00)
        v12_f = self.cnn_v12(v12)
        v24_f = self.cnn_v24(v24)

        fusion_f = torch.concat((v00_f, v12_f, v24_f), dim=1)
        result = self.fusion_linear(fusion_f)

        return result



