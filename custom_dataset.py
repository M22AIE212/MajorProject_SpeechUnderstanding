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
