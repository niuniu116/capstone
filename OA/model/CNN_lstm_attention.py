import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import inception_v3, Inception_V3_Weights

def get_cnn_model(args):
    if args.net_type == "resnet":
        if args.depth == "18":
            model = models.resnet18(pretrained=args.pretrained)
        elif args.depth == "34":
            model = models.resnet34(pretrained=args.pretrained)
        elif args.depth == "50":
            model = models.resnet50(pretrained=args.pretrained)
        elif args.depth == "101":
            model = models.resnet101(pretrained=args.pretrained)
        elif args.depth == "152":
            model = models.resnet152(pretrained=args.pretrained)
        else:
            return None
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, args.feature_dim)

    elif args.net_type == "vgg":
        if args.depth == "16":
            model = models.vgg16(pretrained=args.pretrained)
        elif args.depth == "19":
            model = models.vgg19(pretrained=args.pretrained)
        elif args.depth == "16bn":
            model = models.vgg16_bn(pretrained=args.pretrained)
        elif args.depth == "19bn":
            model = models.vgg19_bn(pretrained=args.pretrained)
        else:
            return None
        num_ftrs = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]
        features.append(nn.Linear(num_ftrs, args.feature_dim))
        model.classifier = nn.Sequential(*features)

    elif args.net_type == "densenet":
        if args.depth == "121":
            model = models.densenet121(pretrained=args.pretrained)
        elif args.depth == "169":
            model = models.densenet169(pretrained=args.pretrained)
        elif args.depth == "201":
            model = models.densenet201(pretrained=args.pretrained)
        else:
            return None
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, args.feature_dim)

    elif args.net_type == "inception":
        if args.depth == "v3":
            model = models.inception_v3(pretrained=args.pretrained, aux_logits=False)
        else:
            return None
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, args.feature_dim)

    else:
        return None

    return model


class CNN_LSTM_ATTENTION(nn.Module):
    def __init__(self, args):
        super(CNN_LSTM_ATTENTION, self).__init__()

        self.single_v = args.single_v
        self.visit_num = args.visit_num
        self.feature_dim = args.feature_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.num_classes = args.num_class

        self.cnn_model = get_cnn_model(args)

        self.time_attn = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
# you can see detial about the lstm in: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Linear(self.lstm_hidden_dim, self.num_classes)

    def forward(self, v00, v12, v24):
        if self.single_v:
            visit_map = {
                'v00': v00,
                'v12': v12,
                'v24': v24
            }
            if self.visit_num not in visit_map:
                raise ValueError(f"Invalid visit number: {self.visit_num}")
            feat = self.cnn_model(visit_map[self.visit_num])
            return self.classifier(feat)

        v00_f = self.cnn_model(v00)
        v12_f = self.cnn_model(v12)
        v24_f = self.cnn_model(v24)

        # [batch_size, 3, feature_dim]
        # Step 1: Stack into sequence: [batch, 3, feature_dim]
        sequence = torch.stack([v00_f, v12_f, v24_f], dim=1)

        # Step 2: Compute attention weights: [batch, 3, 1]
        weights = self.time_attn(sequence)  # [B, 3, 1]
        weights = torch.softmax(weights, dim=1)  # normalize over time axis

        # Step 3: Apply weights
        sequence_weighted = sequence * weights  # [B, 3, D]

        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(sequence_weighted)

        # Use final hidden state
        out = self.classifier(hn[-1])
        return out, weights


