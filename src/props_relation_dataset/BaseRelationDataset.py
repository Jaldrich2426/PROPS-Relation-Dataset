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
from tqdm import tqdm
from timeit import Timer

class BaseRelationDataset(Dataset):

    def __init__(self, split, object_dir, sornet_args, rand_patch, resize):

        """
        
        Args:
            dataset: Dataset objec             
            object_dir: Directory containing object images. Each folder should have the object name, and each image (png) in the folder should have the folder name with a three number index, sequentailly starting at 0
            num_objects: Number of objects to consider in the dataset
            rand_patch: If True, a random object patch is selected for each object. If False, the first object patch is selected for each object"""
        
        # self.obj_id_list
        # self.test_dataset = PROPSPoseDataset(".", "train", download=download)
        self.obj_dir = object_dir
        self.split=split
        self.objects_dict = self._load_objects(obj_dir=object_dir)

        self.rand_patch = rand_patch
        self.max_nobj = sornet_args.max_nobj
        self.resize = resize
        self.img_size = (sornet_args.img_h, sornet_args.img_w)
        self.dataset = self._init_parent_dataset(split,sornet_args)
        # self._load_spatial_relations()
        print("in init")
        self.t = Timer("""x.index(123)""", setup="""x = range(1000)""")
    
    def _get_data(self, idx):
        raise NotImplementedError()
    
    def _get_object_class_list(self):
       raise NotImplementedError()

    def _get_object_images_labels(self):
       raise NotImplementedError()
    
    def _init_parent_dataset(split,sornet_args):
        raise NotImplementedError()
    
    def get_objs_in_image(self,idx):
        """Get object identifiers in image at index idx"""
        raise NotImplementedError()
    

    def get_obj_id(self,obj_name):
        for obj_id, name in self.obj_id_to_name.items():
            if name == obj_name:
                return int(obj_id)
        return None

    def get_obj_idx(self,obj_id_in):
        return obj_id_in-1

    def get_obj_name(self,obj_id):
        return self.obj_id_to_name[obj_id]

    def _load_objects(self, obj_dir):
        # For loop over folders in obj_dir
        obj_dict = {}
        self.obj_id_to_name = {}
        self.obj_idx_to_id = {}
        idx=0
        for obj_type_dir in os.listdir(obj_dir):
            # skip if not a directory
            if not os.path.isdir(os.path.join(obj_dir, obj_type_dir)):
                continue
            obj_id = int(obj_type_dir.split("_")[0])
            obj_idx = idx
            idx+=1
            self.obj_idx_to_id[obj_idx] = obj_id
            # object name is all other parts of the split
            obj_name = "_".join(obj_type_dir.split("_")[1:])
            self.obj_id_to_name[obj_id] = obj_name

            single_obj_type_dict = {}
            for obj_img_file in os.listdir(os.path.join(obj_dir, obj_type_dir)):
                obj_num = int(re.search(r'\d+', obj_img_file).group())
                single_obj_type_dict[obj_num] = Image.open(os.path.join(
                    obj_dir, obj_type_dir, obj_img_file)).convert('RGB')

            # dict to array where key is index
            obj_dict[obj_id] = [single_obj_type_dict[i] for i in range(len(single_obj_type_dict))]
        return obj_dict

    def get_spatial_relations(self, ids,xyz):
        
        # initialize relations list - num relations, num objects, num objects
        # relations_matrix = torch.zeros((4, self.max_nobj, self.max_nobj))
        is_left_of_matrix = torch.ones((self.max_nobj, self.max_nobj))*-1
        is_right_of_matrix = torch.ones((self.max_nobj, self.max_nobj))*-1
        is_front_of_matrix = torch.ones((self.max_nobj, self.max_nobj))*-1
        is_behind_of_matrix = torch.ones((self.max_nobj, self.max_nobj))*-1
        

        # get object relations for a single image
        objs_visib = ids.tolist()

        # for object in the dataset, create a dictionary of the object id and the corresponding pose
        # object pose dictionary in form of object_id: pose
        object_poses = {}
        for objidx, objs_id in enumerate(objs_visib):
            if objs_id == 0:
                continue
            object_poses[objs_id] = xyz[objidx]

        
        for i in range(self.max_nobj):
            obj1 = i+1
            for j in range(self.max_nobj):
                obj2 = j+1
                if obj1 not in object_poses or obj2 not in object_poses:
                    continue
                if i == j:
                    continue
                is_left_of_matrix[i][j] = float(object_poses[obj1][0] < object_poses[obj2][0])
                is_right_of_matrix[i][j] = float(object_poses[obj1][0] > object_poses[obj2][0])
                is_front_of_matrix[i][j] = float(object_poses[obj1][2] < object_poses[obj2][2])
                is_behind_of_matrix[i][j] = float(object_poses[obj1][2] > object_poses[obj2][2])

        # combine relations into 2d array
        relation_matrix = torch.stack([is_left_of_matrix, is_right_of_matrix,
                                      is_front_of_matrix, is_behind_of_matrix], axis=0)
        # print(self.relations.shape)
        return relation_matrix
    


    def __getitem__(self, idx):
        # Get object patches.

        img,ids,xyz = self._get_data(idx)
        
        obj_patches = self._get_obj_patches()
        
        # Get spatial relations.
        relations_matrix = self.get_spatial_relations(ids,xyz)
       
        # combine last two dimensions but exclude diagonal elements between them
        diagonal_mask = torch.ones(self.max_nobj, self.max_nobj, dtype=torch.bool) ^ torch.eye(
            self.max_nobj, dtype=torch.bool)
        relations = relations_matrix[:, diagonal_mask]
     
        # reformat relations matrix
        relations = relations.reshape(-1)

        # Create a mask that will filter out invalid relations.
        mask = self._create_valid_relations_mask(relations)

        # Grab the image as a tensor
        image = torch.tensor(img)

        if(self.resize):
            image = F.resize(image, self.img_size, antialias=True)

        return image, obj_patches, relations.float(), mask.reshape(-1).float()

    def __len__(self):
        return len(self.dataset)

    def _get_obj_patches(self):
        # Generate object patches from the image data.
        obj_patches = []

        # object_images = image_data['object_images']
        for obj_id, pre_patch in self.objects_dict.items():
            patch_idx = 0
            if self.rand_patch:
                patch_idx = torch.randint(len(self.objects_dict[obj_id]), ()).item()

            patch = normalize_rgb(pre_patch[patch_idx])
            obj_patches.append(patch)
        for _ in range(len(obj_patches), self.max_nobj):
            obj_patches.append(torch.zeros_like(obj_patches[0]))
        obj_patches = torch.stack(obj_patches)
        return obj_patches

    def _create_valid_relations_mask(self, relations_matrix):
        # Assuming that the relations matrix has '-1' for invalid relations as in the CLEVR example.
        # has_relations = self.get_spatial_relations(image_data, as_torch=True)

        # Create a mask for valid relations.
        # If '1' indicates a relation and '0' indicates no relation
        mask = relations_matrix != -1

        return mask

    def generate_canidate_patches(self, samples_per_class=12):

        classes = self._get_object_class_list()
        obj_images, obj_labels = self._get_object_images_labels()

        # make object dir if it doesn't exist
        os.makedirs("canidate_objects", exist_ok=True)
        for y, cls in enumerate(classes):
            idxs, = (obj_labels == y).nonzero(as_tuple=True)
            os.makedirs(f"canidate_objects/{y+1}_{cls}", exist_ok=True)

            for i in range(samples_per_class):
                idx = idxs[random.randrange(idxs.shape[0])].item()
                save_image(obj_images[idx], f"canidate_objects/{y+1}_{cls}/{cls}_{i}.png")


normalize_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])