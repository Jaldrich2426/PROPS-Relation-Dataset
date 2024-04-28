from torchvision.utils import make_grid, save_image
import os
import json
from typing import Any, Callable, Optional, Tuple
import random
from torchvision import transforms

import cv2
from PIL import Image

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

from props_relation_dataset.utils import Visualize, chromatic_transform, add_noise
from props_relation_dataset.datasets import PROPSPoseDataset, ProgressObjectsDataset
from props_relation_dataset.utils.data import progress_objects
from torchvision.transforms import functional as F
from pathlib import Path
import re
import argparse
from props_relation_dataset.BaseRelationDataset import BaseRelationDataset

class PROPSRelationDataset(BaseRelationDataset):
    """
    A dataset class for PROPS relation dataset, overriding the BaseRelationDataset class.
    The methods that raise NotImplementedError in the BaseRelationDataset class are implemented here.
    
    Args:
        split (str): The split of the dataset (e.g., 'train', 'val', 'test').
        object_dir (str): The directory containing the object images.
        sornet_args (dict): Arguments for the SORNet model.
        rand_patch (bool): Whether to randomly patch the images.
        resize (tuple): The desired size to resize the images.

    Attributes:
        split (str): The split of the dataset.
        object_dir (str): The directory containing the object images.
        sornet_args (dict): Arguments for the SORNet model.
        rand_patch (bool): Whether to randomly patch the images.
        resize (tuple): The desired size to resize the images.
        dataset (PROPSPoseDataset): The parent dataset.

    Methods:
        _get_data(idx): Get the data for a specific index.
        _get_object_class_list(): Get the list of object classes.
        _get_object_images_labels(): Get the object images and labels.
        _init_parent_dataset(split, sornet_args): Initialize the parent dataset.
        get_objs_in_image(idx): Get the object IDs in an image.
    """

    def __init__(self, split, object_dir, sornet_args, rand_patch, resize):
        """
        Initializes a PropsRelationDataset object.

        Args:
            split (str): The split of the dataset (e.g., 'train', 'val', 'test').
            object_dir (str): The directory containing the object data.
            sornet_args (dict): A dictionary of arguments for the SORNet model.
            rand_patch (bool): Whether to use random patches for training.
            resize (tuple): The desired size to resize the images.

        Returns:
            None
        """
        super().__init__(split, object_dir, sornet_args, rand_patch, resize)

    def _get_data(self, idx):
        """
        Get the data for a specific index.

        Args:
            idx (int): The index of the data to retrieve.

        Returns:
            tuple: A tuple containing the RGB image, object IDs, and RTs.

        """
        datapoint = self.dataset[idx]
        
        # return datapoint info needed
        return datapoint['rgb'], datapoint['objs_id'], datapoint['RTs'][:, :3, 3]

    def _get_object_class_list(self):
            """
            Returns a list of object classes.

            Returns:
                list: A list of object classes.
            """
            classes = [
                "master_chef_can",
                "cracker_box",
                "sugar_box",
                "tomato_soup_can",
                "mustard_bottle",
                "tuna_fish_can",
                "gelatin_box",
                "potted_meat_can",
                "mug",
                "large_marker"
            ]
            return classes

    def _get_object_images_labels(self):
            """
            Retrieves the object images and labels.

            Returns:
                x (list): List of object images.
                y (list): List of object labels.
            """
            x, y, _, _ = progress_objects()
            return x, y

    def _init_parent_dataset(self, split, sornet_args):
        """
        Initializes the parent dataset for the PropsRelationDataset.

        Args:
            split (str): The split of the dataset to use.
            sornet_args: Additional arguments for the parent dataset.

        Returns:
            PROPSPoseDataset: The initialized parent dataset.
        """
        download = not os.path.isdir("PROPS-Pose-Dataset")
        return PROPSPoseDataset(".", split, download=download)

    def get_objs_in_image(self, idx):
            """
            Returns the list of object IDs in the image at the given index.

            Args:
                idx (int): The index of the image.

            Returns:
                list: A list of object IDs in the image.
            """
            return self.dataset[idx]['objs_id']

if __name__ == "__main__":
    # example usage below with dummy args
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--max_nobj', type=int, default=10)
    parser.add_argument('--img_h', type=int, default=320)
    parser.add_argument('--img_w', type=int, default=480)
    parser.add_argument('--n_worker', type=int, default=1)
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--d_hidden', type=int, default=512)
    # Training
    parser.add_argument('--log_dir')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--port', default='12345')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_epoch', type=int, default=80)
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--resume')
    args = parser.parse_args()

    train = PROPSRelationDataset("train","objects",args,rand_patch=True,resize=True)
    # print(train[0])
    train[0]