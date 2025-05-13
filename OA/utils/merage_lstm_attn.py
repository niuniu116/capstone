import torch
import torch.nn as nn
from torchvision import models
import os
import onnx

class Args:
    depth = '19'
    feature_dim = 256
    lstm_hidden_dim = 128
    num_class = 5

def get_cnn_model_fixed(args, checkpoint):
    vgg = models.vgg19(pretrained=False)
    num_ftrs = vgg.classifier[6].in_features
    vgg.classifier[6] = nn.Linear(num_ftrs, args.feature_dim)

    # 加载 cnn 权重
    vgg_state = {k.replace("cnn_model.", ""): v for k, v in checkpoint.items() if k.startswith("cnn_model.")}
    vgg.load_state_dict(vgg_state, strict=False)

    model = nn.Sequential(
        vgg.features,      # (B, 512, 7, 7)
        nn.Flatten(),      # → (B, 25088)
        vgg.classifier     # → (B, 256)
    )
    return model


class CNN_LSTM_Attn(nn.Module):
    def __init__(self, args, checkpoint):
        super(CNN_LSTM_Attn, self).__init__()
        self.feature_dim = args.feature_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.num_class = args.num_class

        self.cnn_model = get_cnn_model_fixed(args, checkpoint)

        self.time_attn = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.lstm = nn.LSTM(
            input_size=args.feature_dim,
            hidden_size=args.lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.classifier = nn.Linear(args.lstm_hidden_dim, args.num_class)

        self.load_remaining_weights(checkpoint)

    def load_remaining_weights(self, checkpoint):
        model_dict = self.state_dict()
        new_state = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(new_state)
        self.load_state_dict(model_dict)

    def forward(self, x):  # x: [B, 3, 3, 224, 224]
        B, V, C, H, W = x.shape
        x = x.view(B * V, C, H, W)
        features = self.cnn_model(x)           # (B*3, 256)
        features = features.view(B, V, -1)     # (B, 3, 256)

        # ===== 时间注意力 =====
        attn_weights = self.time_attn(features)      # (B, 3, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        attended = torch.sum(features * attn_weights, dim=1, keepdim=True)  # (B, 1, 256)

        _, (hn, _) = self.lstm(attended)  # LSTM expects (B, seq, feature)
        out = self.classifier(hn[-1])     # (B, 5)
        return out


def export_to_onnx(pth_path, onnx_path):
    args = Args()

    print("process weights...")
    checkpoint = torch.load(pth_path, map_location="cpu")

    model = CNN_LSTM_Attn(args, checkpoint)
    model.eval()

    print("get ONNX √")
    dummy_input = torch.randn(1, 3, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )

    print(f"successful：{onnx_path}")


if __name__ == "__main__":
    pth_path = r"D:\model\5.14pre\cnn_lstm+attention627041\049-0.625-0.704-0.059.pth"
    onnx_path = r"D:\model\5.14pre\cnn_lstm+attention627041\cnn_lstm_attention.onnx"

    if not os.path.exists(pth_path):
        print("do not exit, check path pls")
    else:
        export_to_onnx(pth_path, onnx_path)
