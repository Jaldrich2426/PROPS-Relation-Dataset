from torchvision.utils import make_grid, save_image
import os
import json
from typing import Any, Callable, Optional, Tuple
import random

import cv2
from PIL import Image

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

from rob599 import Visualize, chromatic_transform, add_noise, PROPSPoseDataset
from rob599.data import progress_objects


class PROPSRelationDataset(PROPSPoseDataset):
    base_folder = "PROPS-Pose-Dataset"
    url = "https://drive.google.com/file/d/15rhwXhzHGKtBcxJAYMWJG7gN7BLLhyAq/view?usp=share_link"
    filename = "PROPS-Pose-Dataset.tar.gz"
    tgz_md5 = "a0c39fe326377dacd1d652f9fe11a7f4"

    def __init__(
        self,
        root: str,
        split: str = 'train',
        download: bool = False,
    ) -> None:
        super().__init__(root, split, download)
 
    def get_spatial_relations(self,image,as_torch=False):

        # get object relations for a single image
        objs_visib = image['objs_id'].tolist()

        # for object in the dataset, create a dictionary of the object id and the corresponding pose
        # object pose dictionary in form of object_id: pose
        object_poses = {}
        for objidx, objs_id in enumerate(objs_visib):
            if objs_id == 0:
                continue
            object_poses[objs_id] = image["RTs"][objidx][:3, 3]

        matrix_size = len(object_poses) *(len(object_poses)-1)
        is_left_of_matrix = np.zeros(matrix_size)
        is_right_of_matrix = np.zeros(matrix_size)
        is_front_of_matrix = np.zeros(matrix_size)
        is_behind_of_matrix = np.zeros(matrix_size)

        num_objects = len(object_poses)

        for i, obj1 in enumerate(object_poses):
            for j, obj2 in enumerate(object_poses):
                if i == j:
                    continue
                is_left_of_matrix[i * (num_objects-1)+j] = object_poses[obj1][0] < object_poses[obj2][0]
                is_right_of_matrix[i * (num_objects - 1) + j] = object_poses[obj1][0] > object_poses[obj2][0]
                is_front_of_matrix[i * (num_objects - 1) + j] = object_poses[obj1][1] < object_poses[obj2][1]
                is_behind_of_matrix[i * (num_objects - 1) + j] = object_poses[obj1][1] > object_poses[obj2][1]
        # combine relations into 2d array
        relations_matrix = np.stack([is_left_of_matrix, is_right_of_matrix, is_front_of_matrix, is_behind_of_matrix], axis=0)
        if as_torch:
            relations_matrix = torch.tensor(relations_matrix)
        return relations_matrix

    def generate_canidate_patches(self, x_train, y_train):
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
        samples_per_class = 12
        # make object dir if it doesn't exist
        os.makedirs("objects", exist_ok=True)
        for y, cls in enumerate(classes):
            idxs, = (y_train == y).nonzero(as_tuple=True)
            for i in range(samples_per_class):
                idx = idxs[random.randrange(idxs.shape[0])].item()
                save_image(x_train[idx], f"objects/sample_{cls}_{i}.png")
       

if __name__ == "__main__":
    # example usage below
    download = not os.path.isdir("PROPS-Pose-Dataset")
    dataset = PROPSRelationDataset(".", "train",download=download)
    # print the relations
    print(dataset.get_spatial_relations(dataset[0],as_torch=True))

    # generate candidate patches
    x_train, y_train, _,_ = progress_objects()
    dataset.generate_canidate_patches(x_train, y_train)
