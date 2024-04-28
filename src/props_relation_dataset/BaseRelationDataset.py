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
        self._load_spatial_relations()

    def _get_object_ids_in_image(self,idx):
        """list of object identifiers in image at given index"""
        raise NotImplementedError()

    def _get_object_xyzs_in_image(self,idx,obj_idx):
        """
        2d array of object x,y,z coordinates in image
        first dimension is object index, second is x,y,z
        idx: index of image in dataset
        obj_idx: index of object in image wrt the object dictionary
        """
        raise NotImplementedError()

    def _get_image(self,idx):
        """Get rgb image at index idx"""
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

    def _load_spatial_relations(self):
        if (os.path.exists(os.path.join(self.obj_dir, f"{self.split}_relations.pt"))):
            self.relations = torch.load(os.path.join(self.obj_dir, f"{self.split}_relations.pt"))
            return
        # initialize relations list - 4d tensor or len dataset, num relations, num objects, num objects
        self.relations = torch.zeros((len(self.dataset), 4, self.max_nobj, self.max_nobj))
        is_left_of_matrix = torch.ones((len(self.dataset), self.max_nobj, self.max_nobj))*-1
        is_right_of_matrix = torch.ones((len(self.dataset), self.max_nobj, self.max_nobj))*-1
        is_front_of_matrix = torch.ones((len(self.dataset), self.max_nobj, self.max_nobj))*-1
        is_behind_of_matrix = torch.ones((len(self.dataset), self.max_nobj, self.max_nobj))*-1
        print("Generating and saving spatial relations")
        for idx in tqdm(range(len(self.dataset))):
            # get object relations for a single image
            objs_visib = self._get_object_ids_in_image(idx)

            # for object in the dataset, create a dictionary of the object id and the corresponding pose
            # object pose dictionary in form of object_id: pose
            object_poses = {}
            for objidx, objs_id in enumerate(objs_visib):
                if objs_id == 0:
                    continue
                object_poses[objs_id] = self._get_object_xyzs_in_image(idx,objidx)

            # fill to -1
            # is_left_of_matrix = torch.ones((self.max_nobj, self.max_nobj))*-1
            # is_right_of_matrix = torch.ones((self.max_nobj, self.max_nobj))*-1
            # is_front_of_matrix = torch.ones((self.max_nobj, self.max_nobj))*-1
            # is_behind_of_matrix = torch.ones((self.max_nobj, self.max_nobj))*-1

            # num_objects = len(object_poses)
            # print(len(object_poses))
            
        

            for i in range(self.max_nobj):
                obj1 = i+1
                for j in range(self.max_nobj):
                    obj2 = j+1
                    if obj1 not in object_poses or obj2 not in object_poses:
                        continue
                    if i == j:
                        continue
                    is_left_of_matrix[idx][i][j] = float(object_poses[obj1][0] < object_poses[obj2][0])
                    is_right_of_matrix[idx][i][j] = float(object_poses[obj1][0] > object_poses[obj2][0])
                    is_front_of_matrix[idx][i][j] = float(object_poses[obj1][2] < object_poses[obj2][2])
                    is_behind_of_matrix[idx][i][j] = float(object_poses[obj1][2] > object_poses[obj2][2])

        # combine relations into 2d array
        self.relations = torch.stack([is_left_of_matrix, is_right_of_matrix,
                                        is_front_of_matrix, is_behind_of_matrix], axis=1)
        torch.save(self.relations, os.path.join(self.obj_dir, f"{self.split}_relations.pt"))
        # print(self.relations.shape)
        # print(self.relations[idx].shape)
        # print(relation_matrix.shape)
        # self.relations= relation_matrix
        # relations_matrix = torch.tensor(relations_matrix)

    def get_spatial_relations(self, idx):
        return self.relations[idx]

    def __getitem__(self, idx):
        # Get object patches.
        obj_patches = self._get_obj_patches()

        # Get spatial relations.
        relations_matrix = self.get_spatial_relations(idx)

        # combine last two dimensions but exclude diagonal elements between them
        diagonal_mask = torch.ones(self.max_nobj, self.max_nobj, dtype=torch.bool) ^ torch.eye(
            self.max_nobj, dtype=torch.bool)
        relations = relations_matrix[:, diagonal_mask]

        # reformat relations matrix
        relations = relations.reshape(-1)

        # Create a mask that will filter out invalid relations.
        mask = self._create_valid_relations_mask(relations)

        # Grab the image as a tensor
        image = torch.tensor(self._get_image(idx))

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



# class TO_Scene_Rel(BaseRelationDataset):
#     def __init__(self, split, object_dir, sornet_args, rand_patch, resize):
#         super().__init__(split, object_dir, sornet_args, rand_patch, resize)

#     def _get_object_ids_in_image(self, idx):
#         return self.dataset[idx]['objs_id']

#     def _get_object_xyzs_in_image(self, idx, obj_idx):
#         return self.dataset[idx]["RTs"][obj_idx][:3, 3]

#     def _get_image(self, idx):
#         return self.dataset[idx]['rgb']

#     def _get_object_class_list(self):
#         classes = [
#             "master_chef_can",
#             "cracker_box",
#             "sugar_box",
#             "tomato_soup_can",
#             "mustard_bottle",
#             "tuna_fish_can",
#             "gelatin_box",
#             "potted_meat_can",
#             "mug",
#             "large_marker"
#         ]
#         return classes

#     def _get_object_images_labels(self):
#         x, y, _, _ = progress_objects()
#         return x, y

#     def _init_parent_dataset(self, split, sornet_args):
#         download = not os.path.isdir("PROPS-Pose-Dataset")
#         return PROPSPoseDataset(".", split, download=download)

# if __name__ == "__main__":
#     # example usage below with dummy args
#     parser = argparse.ArgumentParser()
#     # Data
#     parser.add_argument('--max_nobj', type=int, default=10)
#     parser.add_argument('--img_h', type=int, default=320)
#     parser.add_argument('--img_w', type=int, default=480)
#     parser.add_argument('--n_worker', type=int, default=1)
#     # Model
#     parser.add_argument('--patch_size', type=int, default=32)
#     parser.add_argument('--width', type=int, default=768)
#     parser.add_argument('--layers', type=int, default=12)
#     parser.add_argument('--heads', type=int, default=12)
#     parser.add_argument('--d_hidden', type=int, default=512)
#     # Training
#     parser.add_argument('--log_dir')
#     parser.add_argument('--n_gpu', type=int, default=1)
#     parser.add_argument('--port', default='12345')
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--lr', type=float, default=0.0001)
#     parser.add_argument('--n_epoch', type=int, default=80)
#     parser.add_argument('--print_freq', type=int, default=5)
#     parser.add_argument('--eval_freq', type=int, default=1)
#     parser.add_argument('--save_freq', type=int, default=1)
#     parser.add_argument('--resume')
#     args = parser.parse_args()

#     train = PROPS_REL("train","objects",args,rand_patch=True,resize=True)

#     # download = not os.path.isdir("PROPS-Pose-Dataset")
#     # dataset = PROPSPoseDataset(".", "train", download=download)

#     # # print(dataset[0]['objs_id'])
#     # sornet_data = PROPS_REL(dataset,"objects",10,True,False,(320,480))
#     # # for i in range(500):
#     # image, obj_patches, relations, mask = sornet_data[0]
#     # # print(sornet_data.obj_id_to_name)
#     # x, y, _, _ = progress_objects()
    
#     # # make object dir if it doesn't exist
#     # sornet_data.generate_canidate_patches(12)
#     # print()
#     # print(relations.shape)
#     # print(sornet_data)
#     # print(sornet_data.get_spatial_relations(dataset[0]).shape)
#     # obj_counts = torch.zeros(11)
#     # for i in range(500):
    #     data_dict = dataset[i]
    #     # print(torch.tensor(data_dict['objs_id']))
    #     # obj_counts[torch.tensor(data_dict['objs_id'])] += 1
    #     # print(i)
    #     for obj_id in data_dict['objs_id']:
    #         # print(obj_id)
    #         obj_counts[obj_id] += 1

    # dataset_val = PROPSPoseDataset(".", "train", download=download)
    # for i in range(500):
    #     data_dict = dataset_val[i]
    #     # obj_counts[data_dict['objs_id']] += 1
    #     for obj_id in data_dict['objs_id']:

    #         obj_counts[obj_id] += 1
    # # print(obj_counts)
