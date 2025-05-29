# -*- coding: utf-8 -*-

import os, sys, pdb
import torch.utils.data as data

from PIL import Image
import os
import os.path
from PIL import ImageFilter
import numpy as np
import pandas as pd
import torch
import cv2
from einops import repeat

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


# def find_classes(dir):
#     classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
#     classes.sort()
#     class_to_idx = {classes[i]: i for i in range(len(classes))}
#     return classes, class_to_idx

def make_dataset(root):
    kl_grades = [0, 1, 2, 3, 4]
    images = []
    for kl_grade in kl_grades:
        images_path = os.path.join(root, str(kl_grade))
        patients = os.listdir(images_path)
        for patient in patients:
            patient_path = os.path.join(images_path, patient)
            img_list = []
            for image in os.listdir(patient_path):
                img_list.append(os.path.join(patient_path, image))
            images.append((img_list, kl_grade))
    return images

# def make_dataset(dir, class_to_idx):
#     images = []
#     dir = os.path.expanduser(dir)
#     for target in sorted(os.listdir(dir)):
#         d = os.path.join(dir, target)
#         if not os.path.isdir(d):
#             continue
#
#         for root, _, fnames in sorted(os.walk(d)):
#             for fname in sorted(fnames):
#                 if is_image_file(fname):
#                     path = os.path.join(root, fname)
#                     #patient = fname.split('.')[0]
#                     #if patient in patient_list:
#                     item = (path, class_to_idx[target])
#                         #item = (path, int(patient_kl[patient].tolist()))
#                     images.append(item)
#
#     return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        # img = img.resize((224, 224), Image.BICUBIC)
        # img = img.filter(ImageFilter.FIND_EDGES)
        # img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        return img


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None, loader=pil_loader):
        # classes, class_to_idx = find_classes(root)

        # imgs = make_dataset(root, class_to_idx)

        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        # self.classes = classes
        # self.class_to_idx = class_to_idx
        self.classes = ['0', '1', '2', '3', '4']
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader




    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # img, target, patientname = self.imgs[index]
        paths, target = self.imgs[index]
        img_1 = self.image_load(paths[0])
        img_2 = self.image_load(paths[1])
        img_3 = self.image_load(paths[2])
        # img = torch.concat((img_1, img_2, img_3), 1)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # filename = os.path.basename(path)
        # filename = filename.split('.')[0].split('_')[0]
        # print(img.shape)
        return img_1, img_2, img_3, target

    def __len__(self):
        return len(self.imgs)

    def image_load(self, path):
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)
        # img = repeat(img, 'c h w -> c t h w', t=1)
        return img


