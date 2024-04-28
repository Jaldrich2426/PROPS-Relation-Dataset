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
# from numpy import load
from plyfile import PlyData, PlyElement
from open3d import *

class PROPSRelationDataset(BaseRelationDataset):
    def __init__(self, split, object_dir, sornet_args, rand_patch, resize):
        super().__init__(split, object_dir, sornet_args, rand_patch, resize)

    


    def _get_data(self, idx):
        datapoint=self.dataset[idx]
        
        # return datapoint info needed
        return datapoint['rgb'], datapoint['objs_id'], datapoint['RTs'][:, :3, 3]

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

if __name__ == "__main__":
    # example usage below with dummy args
    # parser = argparse.ArgumentParser()
    # # Data
    # parser.add_argument('--max_nobj', type=int, default=10)
    # parser.add_argument('--img_h', type=int, default=320)
    # parser.add_argument('--img_w', type=int, default=480)
    # parser.add_argument('--n_worker', type=int, default=1)
    # # Model
    # parser.add_argument('--patch_size', type=int, default=32)
    # parser.add_argument('--width', type=int, default=768)
    # parser.add_argument('--layers', type=int, default=12)
    # parser.add_argument('--heads', type=int, default=12)
    # parser.add_argument('--d_hidden', type=int, default=512)
    # # Training
    # parser.add_argument('--log_dir')
    # parser.add_argument('--n_gpu', type=int, default=1)
    # parser.add_argument('--port', default='12345')
    # parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--n_epoch', type=int, default=80)
    # parser.add_argument('--print_freq', type=int, default=5)
    # parser.add_argument('--eval_freq', type=int, default=1)
    # parser.add_argument('--save_freq', type=int, default=1)
    # parser.add_argument('--resume')
    # args = parser.parse_args()

    # train = PROPSRelationDataset("train","objects",args,rand_patch=True,resize=True)
    # # print(train[0])
    # train[0]

    # data = np.load("TO-vanilla/npz/train/id0.npz")
    # lst=data.files
    # for item in lst:
    #     print(item)
    #     print(data[item])
    cloud = io.read_point_cloud("lm/models/obj_000001.ply")  # Read point cloud
    visualization.draw_geometries([cloud])    # Visualize point cloud
