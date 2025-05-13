import numpy as np
from PIL import Image
import os

img_paths = [
    r"D:\Backbone_3\v3_data\train\0\9003658_1\9003658_1_v00m.png",
    r"D:\Backbone_3\v3_data\train\0\9003658_1\9003658_1_v12m.png",
    r"D:\Backbone_3\v3_data\train\0\9003658_1\9003658_1_v24m.png"
]

imgs = []
for path in img_paths:
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0  # HWC
    arr = arr.transpose(2, 0, 1)  # → CHW
    imgs.append(arr)

# (3, 3, 224, 224) → batch_size = 1
merged = np.stack(imgs, axis=0)
merged = np.stack(imgs, axis=0)
merged = np.expand_dims(merged, axis=0)

# save .npy
out_path = r"D:\Backbone_3\final_input.npy"
np.save(out_path, merged)
print(f"successful：{out_path}")
