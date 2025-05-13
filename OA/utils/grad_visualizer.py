# grad_visualizer.py
import shutil
import os
import torch
import numpy as np
import cv2
from torchvision.utils import save_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt


def get_target_layer(model, args):
    if args.net_type.startswith("vgg"):
        return model.features[-1]
    elif args.net_type.startswith("resnet"):
        return model.layer4[-1]
    elif args.net_type.startswith("densenet"):
        return model.features[-1]
    elif args.net_type.startswith("inception"):
        return model.Mixed_7c
    else:
        raise ValueError("pls set suitable target_layer")


def visualize_gradcam(model, dataloader, args, epoch, output_dir="vis_results"):
    if not args.visualize or (epoch != 0 and epoch % args.visualize_interval != 0):
        return
    if epoch == 0 and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    for i, batch in enumerate(dataloader):
        v00, v12, v24, labels = [x.to(device) for x in batch]
        v00.requires_grad_()
        v12.requires_grad_()
        v24.requires_grad_()

        if hasattr(model, "cnn_model"):
            cnn_model = model.cnn_model.to(device)
            visit_map = {"v00": v00, "v12": v12, "v24": v24}
            input_img = visit_map.get(args.visit_num, v00)
        elif hasattr(model, "cnn_v00") and hasattr(model, "cnn_v12") and hasattr(model, "cnn_v24"):
            visit_cnn_map = {
                "v00": (model.cnn_v00, v00),
                "v12": (model.cnn_v12, v12),
                "v24": (model.cnn_v24, v24),
            }
            cnn_model, input_img = visit_cnn_map.get(args.visit_num, (model.cnn_v00, v00))
            cnn_model = cnn_model.to(device)
        else:
            raise AttributeError("The model does not contain cnn_model or cnn_vxx structures and cannot be used for Grad-CAM visualization")

        target_layer = get_target_layer(cnn_model, args)

        with torch.no_grad():
            model_output = model(v00, v12, v24)

        if isinstance(model_output, tuple) and len(model_output) == 2:
            outputs, weights = model_output
        else:
            outputs = model_output
            weights = torch.ones(v00.size(0), 3).to(v00.device) / 3

        for j in range(v00.size(0)):
            sample_idx = i * dataloader.batch_size + j
            att = weights[j].squeeze().detach().cpu().numpy()

            for t_idx, (v_tensor, tag) in enumerate(zip([v00, v12, v24], ['v00', 'v12', 'v24'])):
                input_img = v_tensor[j].unsqueeze(0)
                with GradCAM(model=cnn_model, target_layers=[target_layer]) as cam:
                    grayscale_cam = cam(input_tensor=input_img, targets=None)[0]

                rgb_img = input_img.squeeze().detach().cpu().permute(1, 2, 0).numpy()
                rgb_img = np.clip((rgb_img * 0.229 + 0.485), 0, 1)
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                cam_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                epoch_dir = os.path.join(output_dir, f"epoch_{epoch}", f"sample_{sample_idx}")
                os.makedirs(epoch_dir, exist_ok=True)

                save_image(input_img, f"{epoch_dir}/{tag}_input.jpg")
                cv2.imwrite(f"{epoch_dir}/{tag}_cam.jpg", cam_bgr)

            np.save(f"{epoch_dir}/attention_weights.npy", att)
            draw_attention_bar(att, f"{epoch_dir}/attention_plot.png")

        break



def draw_attention_bar(att, save_path, labels=["v00", "v12", "v24"]):
    plt.figure(figsize=(4, 3))
    plt.bar(labels, att, color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel("Attention Weight")
    plt.title("Temporal Attention")
    for i, v in enumerate(att):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
