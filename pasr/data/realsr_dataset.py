import cv2
import os
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.degradations import add_jpg_compression
from basicsr.data.transforms import augment, mod_crop, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class RealSRDataset(data.Dataset):

    def __init__(self, opt):
        super(RealSRDataset, self).__init__()
        self.opt = opt

    def __getitem__(self, index):
        return {'lq': None, 'gt': None, 'lq_path': None, 'gt_path': None}

    def __len__(self):
        return 0
