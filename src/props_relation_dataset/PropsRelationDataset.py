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
    def __init__(self, split, object_dir, sornet_args, rand_patch, resize):
        super().__init__(split, object_dir, sornet_args, rand_patch, resize)

    def _get_object_ids_in_image(self, idx):
        return self.dataset[idx]['objs_id']

    def _get_object_xyzs_in_image(self, idx, obj_idx):
        return self.dataset[idx]["RTs"][obj_idx][:3, 3]

    def _get_image(self, idx):
        return self.dataset[idx]['rgb']

    def _get_object_class_list(self):
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
        x, y, _, _ = progress_objects()
        return x, y

    def _init_parent_dataset(self, split, sornet_args):
        download = not os.path.isdir("PROPS-Pose-Dataset")
        return PROPSPoseDataset(".", split, download=download)

    def get_objs_in_image(self, idx):
        return self.dataset[idx]['objs_id']
