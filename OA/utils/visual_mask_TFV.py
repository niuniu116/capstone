import os
import torch
import shutil
from torchvision.utils import save_image
from einops import rearrange
from torch.autograd import Variable

def mask(x, idx, patch_size):
    """
    Args:
        x: input image, shape: [B, 3, H, W]
        idx: indices of masks, shape: [B, T], value in range [0, h*w)
    Return:
        out_img: masked image with only patches from idx postions
    """
    h = x.size(2) // patch_size
    x = rearrange(x, 'b c (h p) (w q) -> b (c p q) (h w)', p=patch_size, q=patch_size)
    output = torch.zeros_like(x)
    idx1 = idx.unsqueeze(1).expand(-1, x.size(1), -1)
    extracted = torch.gather(x, dim=2, index=idx1)  # [b, c p q, T]
    scattered = torch.scatter(output, dim=2, index=idx1, src=extracted)
    out_img = rearrange(scattered, 'b (c p q) (h w) -> b c (h p) (w q)', p=patch_size, q=patch_size, h=h)
    return out_img


def get_deeper_idx(idx1, idx2):
    """
    Args:
        idx1: indices, shape: [B, T1]
        idx2: indices to gather from idx1, shape: [B, T2], T2 <= T1
    """
    return torch.gather(idx1, dim=1, index=idx2)


def get_real_idx(idxs, fuse_token):
    # nh = img_size // patch_size
    # npatch = nh ** 2

    # gather real idx
    for i in range(1, len(idxs)):
        tmp = idxs[i - 1]
        if fuse_token:
            B = tmp.size(0)
            tmp = torch.cat([tmp, torch.zeros(B, 1, dtype=tmp.dtype, device=tmp.device)], dim=1)
        idxs[i] = torch.gather(tmp, dim=1, index=idxs[i])
    return idxs


def save_img_batch(x, path, file_name='img{}', start_idx=0):
    for i, img in enumerate(x):
        save_image(img, os.path.join(path, file_name.format(start_idx + i)))

def visualize_mask(data_loader, model, device, output_dir, fuse_token, args):
    model.eval()
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    ii = 0
    num_batch = 0
    for data in data_loader:

        v00_inputs, v12_inputs, v24_inputs, labels = data
        v00_inputs, v12_inputs, v24_inputs = Variable(v00_inputs.to(device=device)), Variable(
            v12_inputs.to(device=device)), Variable(v24_inputs.to(device=device))

        B = v00_inputs.size(0)

        with torch.cuda.amp.autocast():
            if args.model_type == 'vit_token_fusion' and args.visit_num == 'v00':
                images = v00_inputs
                outputs, idx = model(v00_inputs, args.base_keep_rate, get_idx=True)
            elif args.model_type == 'vit_token_fusion' and args.visit_num == 'v12':
                images = v12_inputs
                outputs, idx = model(v12_inputs, args.base_keep_rate, get_idx=True)
            elif args.model_type == 'vit_token_fusion' and args.visit_num == 'v24':
                images = v24_inputs
                outputs, idx = model(v24_inputs, args.base_keep_rate, get_idx=True)

        idxs = get_real_idx(idx, fuse_token)
        for jj, idx in enumerate(idxs):
            masked_img = mask(images, patch_size=16, idx=idx)
            save_img_batch(masked_img, output_dir, file_name=f'img_{num_batch}' + '_{}' + f'_l{jj}.jpg')

        save_img_batch(images, output_dir, file_name=f'img_{num_batch}' + '_{}_a.jpg')
        ii += 1
        num_batch += 1
