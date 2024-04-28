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
from timeit import Timer

class PROPSRelationDataset(Dataset):
    
    def __init__(self,split: str = 'train'):
        
        assert split in ['train', 'val']
        download = not os.path.isdir("PROPS-Pose-Dataset")
        self.dataset = PROPSPoseDataset(".", split, download=download)
        # self.test_dataset = PROPSPoseDataset(".", "train", download=download)
        self.objects_dict = self._load_objects("objects")
        self.rand_patch = False
        self.max_nobj = 10
        self.t = Timer("""x.index(123)""", setup="""x = range(1000)""")

    def _load_objects(self,obj_dir):
        # For loop over folders in obj_dir
        obj_dict={}
        for obj_type_dir in os.listdir(obj_dir):
            # object_name = obj_type.name
            obj_id=obj_type_dir.split("_")[0]
            

            single_obj_type_dict = {}
            for obj_img_file in os.listdir(os.path.join(obj_dir,obj_type_dir)):
                obj_num = int(re.search(r'\d+', obj_img_file).group())
                single_obj_type_dict[obj_num] = Image.open(os.path.join(obj_dir,obj_type_dir,obj_img_file)).convert('RGB')

            # dict to array where key is index
            obj_dict[obj_id] = [single_obj_type_dict[i] for i in range(len(single_obj_type_dict))]
        return obj_dict
                
    def get_spatial_relations(self, image):

        # get object relations for a single image
        objs_visib = image['objs_id'].tolist()

        # for object in the dataset, create a dictionary of the object id and the corresponding pose
        # object pose dictionary in form of object_id: pose
        object_poses = {}
        for objidx, objs_id in enumerate(objs_visib):
            if objs_id == 0:
                continue
            object_poses[objs_id] = image["RTs"][objidx][:3, 3]

        matrix_size = self.max_nobj * (self.max_nobj-1)
        # fill to -1
        # is_left_of_matrix = -np.ones(matrix_size)
        # is_right_of_matrix = -np.ones(matrix_size)
        # is_front_of_matrix = -np.ones(matrix_size)
        # is_behind_of_matrix = -np.ones(matrix_size)

        is_left_of_matrix = torch.ones((self.max_nobj,self.max_nobj))*-1
        is_right_of_matrix = torch.ones((self.max_nobj, self.max_nobj))*-1
        is_front_of_matrix = torch.ones((self.max_nobj, self.max_nobj))*-1
        is_behind_of_matrix = torch.ones((self.max_nobj, self.max_nobj))*-1


        num_objects = len(object_poses)
        # print(len(object_poses))
        for i in range(self.max_nobj):
            obj1=i+1
            for j in range(self.max_nobj):
                obj2=j+1
                if obj1 not in object_poses or obj2 not in object_poses:
                    continue
        # for i, obj1 in enumerate(object_poses):
        #     # print(obj1)
        #     for j, obj2 in enumerate(object_poses):
                if i == j:
                    # is_left_of_matrix[i][j] = -2
                    # is_right_of_matrix[i][j] = -2
                    # is_front_of_matrix[i][j] = -2
                    # is_behind_of_matrix[i][j] = -2
                    continue
                # is_left_of_matrix[i * (num_objects-1)+j] = object_poses[obj1][0] < object_poses[obj2][0]
                # is_right_of_matrix[i * (num_objects - 1) + j] = object_poses[obj1][0] > object_poses[obj2][0]
                # is_front_of_matrix[i * (num_objects - 1) + j] = object_poses[obj1][1] < object_poses[obj2][1]
                # is_behind_of_matrix[i * (num_objects - 1) + j] = object_poses[obj1][1] > object_poses[obj2][1]
                is_left_of_matrix[i][j] = float(object_poses[obj1][0] < object_poses[obj2][0])
                is_right_of_matrix[i][j] = float(object_poses[obj1][0] > object_poses[obj2][0])
                is_front_of_matrix[i][j] = float(object_poses[obj1][2] < object_poses[obj2][2])
                is_behind_of_matrix[i][j] = float(object_poses[obj1][2] > object_poses[obj2][2])
        # combine relations into 2d array
        relations_matrix = torch.stack([is_left_of_matrix, is_right_of_matrix,
                                    is_front_of_matrix, is_behind_of_matrix], axis=0)
        
        # relations_matrix = torch.tensor(relations_matrix)
        return relations_matrix

    def __getitem__(self, idx):
        
        data_dict = self.dataset[idx]
        # print(self.t.timeit(0))
        # Get object patches.
        obj_patches = self._get_obj_patches()
        # print(self.t.timeit(0))

        # Calculate relations -3D matrix, first dimension is relation type, second and third are object indices
        relations_matrix = self.get_spatial_relations(data_dict)
        # print(self.t.timeit(0))

        # reformat relations matrix

        # combine last two dimensions but exclude diagonal elements between them
        diagonal_mask = torch.ones(self.max_nobj,self.max_nobj,dtype=torch.bool) ^ torch.eye(self.max_nobj, dtype=torch.bool)
        # print(self.t.timeit(0))
        # print(diagonal_mask)
        relations = relations_matrix[:, diagonal_mask]
        # print(self.t.timeit(0))

        # relations = relations_matrix.reshape(4, -1)
        # print(relations[0] == -2)
        # relations = relations[:, relations[0] != -2]
        
        # flatten
        relations = relations.reshape(-1)
        # print(self.t.timeit(0))

        # Create a mask that will filter out invalid relations.

        mask = self._create_valid_relations_mask(relations)
        # print(self.t.timeit(0))
        image = torch.tensor(data_dict['rgb'])
        # print(image.shape)
        # print(image.shape)
        # resize image to 480x320 (width x height)
        # print(self.t.timeit(0))
        # print(image.shape)
        image = F.resize(image, (480, 640),antialias=True)
        # print(relations_matrix.dtype)
        # print(mask.dtype)
        # print(mask.reshape(-1).float().dtype)
        # print(self.t.timeit(0))
        # print(obj_patches.dtype)
        # print(mask.reshape(-1).shape)
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
        
            patch=normalize_rgb(pre_patch[patch_idx])
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

    def generate_canidate_patches(self):
        x_train, y_train, _, _ = progress_objects()
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
        os.makedirs("canidate_objects", exist_ok=True)
        for y, cls in enumerate(classes):
            idxs, = (y_train == y).nonzero(as_tuple=True)
            for i in range(samples_per_class):
                idx = idxs[random.randrange(idxs.shape[0])].item()
                save_image(x_train[idx], f"canidate_objects/sample_{cls}_{i}.png")
    
normalize_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

if __name__ == "__main__":
    # example usage below
    download = not os.path.isdir("PROPS-Pose-Dataset")
    dataset = PROPSPoseDataset(".", "train", download=download)

    # print(dataset[0]['objs_id'])
    sornet_data = PROPSRelationDataset()
    # for i in range(500):
    image, obj_patches, relations, mask =sornet_data[0]
    # print(relations.shape)
    # print(sornet_data)
    # print(sornet_data.get_spatial_relations(dataset[0]).shape)
    # obj_counts = torch.zeros(11)
    # for i in range(500):
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
        

