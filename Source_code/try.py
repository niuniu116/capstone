from torchvision import models
import torch
import torch
import torch.nn as nn
from torchvision import models



# checkpoint = torch.hub.load_state_dict_from_url(
#     'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth', map_location='cpu', check_hash=True)
# print(checkpoint['model'].keys())
# print(checkpoint['model']['pos_embed'].shape[-1])

model = torch.load('data/model/vgg_v00/006-0.594-0.601-0.031.pth')
print(model.keys())