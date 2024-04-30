import random
from PIL import Image
from PIL import ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Dataset class for Siamese Network.

    Args:
    - imageFolderDataset: Dataset containing image folders
    - transform: Transformation to apply to the images
    - should_invert: Whether to invert the images
    """

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = ImageOps.invert(img0)
            img1 = ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        labels = torch.squeeze(torch.tensor([int(img1_tuple[1] == img0_tuple[1])], dtype=torch.float32)).long()
        # Label = 1 when the images are from the same class
        return img0, img1, labels

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
