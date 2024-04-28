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
    """
    Base class for relation datasets.

    Args:
        split (str): The split of the dataset (e.g., 'train', 'val', 'test').
        object_dir (str): Directory containing object images.
        sornet_args: Arguments for the SORNet model.
        rand_patch (bool): If True, a random object patch is selected for each object. If False, the first object patch is selected for each object.
        resize (bool): If True, resize the image to the specified size.

    Attributes:
        obj_dir (str): Directory containing object images.
        split (str): The split of the dataset.
        objects_dict (dict): Dictionary mapping object IDs to a list of object images.
        rand_patch (bool): If True, a random object patch is selected for each object. If False, the first object patch is selected for each object.
        max_nobj (int): Maximum number of objects in the dataset.
        resize (bool): If True, resize the image to the specified size.
        img_size (tuple): Image size (height, width).
        dataset: The parent dataset.

    Methods:
        _get_data(idx): Abstract method to get data for a given index.
        _get_object_class_list(): Abstract method to get the list of object classes.
        _get_object_images_labels(): Abstract method to get the object images and labels.
        _init_parent_dataset(split, sornet_args): Abstract method to initialize the parent dataset.
        get_objs_in_image(idx): Get object identifiers in the image at the given index.
        get_obj_id(obj_name): Get the object ID for the given object name.
        get_obj_idx(obj_id_in): Get the object index for the given object ID.
        get_obj_name(obj_id): Get the object name for the given object ID.
        _load_objects(obj_dir): Load the object images from the given directory.
        get_spatial_relations(ids, xyz): Get the spatial relations between objects.
        __getitem__(idx): Get the data for the given index.
        __len__(): Get the length of the dataset.
        _get_obj_patches(): Generate object patches from the image data.
        _create_valid_relations_mask(relations_matrix): Create a mask for valid relations.
        generate_canidate_patches(samples_per_class): Generate candidate object patches for each class.
    """
    def __init__(self, split, object_dir, sornet_args, rand_patch, resize):
        """
        Initializes the BaseRelationDataset object.

        Args:
            split (str): The split of the dataset (e.g., "train", "val", "test").
            object_dir (str): The directory path where the object files are stored.
            sornet_args (object): An object containing arguments for the SORNet model.
            rand_patch (bool): A flag indicating whether to apply random patch augmentation.
            resize (bool): A flag indicating whether to resize the images.

        Returns:
            None
        """
        self.obj_dir = object_dir
        self.split = split
        self.objects_dict = self._load_objects(obj_dir=object_dir)
        self.rand_patch = rand_patch
        self.max_nobj = sornet_args.max_nobj
        self.resize = resize
        self.img_size = (sornet_args.img_h, sornet_args.img_w)
        self.dataset = self._init_parent_dataset(split, sornet_args)
    
    def _get_data(self, idx):
            """
            Retrieves the data for a given index. 

            This method should be implemented by subclasses to provide the data for the given index.

            Parameters:
                idx (int): The index of the data to retrieve.

            Returns:
                tuple: A tuple containing the RGB image, object IDs, and XYZ coordinates.
            """
            raise NotImplementedError()
            # return rgb, obj_ids, xyz
    
    def _get_object_class_list(self):
            """
            Returns a list of object classes.

            This method should be implemented by subclasses to provide the list of object classes
            specific to their dataset.

            Returns:
                list: A list of object classes.
            """
            raise NotImplementedError()
            # return classes

    def _get_object_images_labels(self):
            """
            Retrieves the images and labels of the objects in the dataset.

            This method should be implemented by subclasses to provide the images and labels of the objects.

            Returns:
                obj_images (list): A list of images of the objects.
                obj_labels (list): A list of labels corresponding to the objects.
            """
            raise NotImplementedError()
            # return obj_images, obj_labels
    
    def _init_parent_dataset(split, sornet_args):
        """
        Initializes the parent dataset.

        This method should be implemented by subclasses to initialize the parent dataset.

        Args:
            split (str): The split of the dataset.
            sornet_args (dict): Arguments for the Sornet model.

        Returns:
            parent_dataset: The initialized parent dataset.
        """
        raise NotImplementedError()
        # return parent_dataset
    
    def get_objs_in_image(self, idx):
        """
        Returns the objects in the image at the given index.

        This method should be implemented by subclasses to return the objects in the image at the given index.

        Parameters:
        - idx (int): The index of the image.

        Returns:
        - list: A list of object IDs present in the image.
        """
        raise NotImplementedError()
        # return objs_id
    
    def get_obj_id(self, obj_name):
            """
            Returns the object ID corresponding to the given object name.

            Args:
                obj_name (str): The name of the object.

            Returns:
                int or None: The object ID if found, None otherwise.
            """
            for obj_id, name in self.obj_id_to_name.items():
                if name == obj_name:
                    return int(obj_id)
            return None

    def get_obj_idx(self, obj_id_in):
        """
        Returns the index of the object based on its ID. Assumes that object IDs are 1-indexed but can be overridden.

        Parameters:
        obj_id_in (int): The ID of the object.

        Returns:
        int: The index of the object.
        """
        return obj_id_in - 1

    def get_obj_name(self, obj_id):
            """
            Returns the name of the object corresponding to the given object ID.

            Args:
                obj_id (int): The ID of the object.

            Returns:
                str: The name of the object.

            Raises:
                KeyError: If the object ID is not found in the obj_id_to_name dictionary.
            """
            return self.obj_id_to_name[obj_id]

    def _load_objects(self, obj_dir):
        """
        Load objects from the specified directory.

        Args:
            obj_dir (str): The directory path where the objects are stored.

        Returns:
            dict: A dictionary containing the loaded objects, where the keys are object IDs and the values are lists of images.
        """
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

    def get_spatial_relations(self, ids, xyz):
        """
        Get the spatial relations between objects based on their positions.

        Args:
            ids (torch.Tensor): Tensor containing the object IDs.
            xyz (torch.Tensor): Tensor containing the object positions.

        Returns:
            torch.Tensor: Tensor containing the spatial relations between objects.
                The shape of the tensor is (4, max_nobj, max_nobj), where:
                - The first dimension represents the types of spatial relations:
                    - Index 0: is_left_of
                    - Index 1: is_right_of
                    - Index 2: is_front_of
                    - Index 3: is_behind_of
                - The second and third dimensions represent the object indices.
                - The values in the tensor indicate the presence (1.0), absence (0.0), or invalidity (-1.0) of the spatial relation.
        """
        # initialize relations list - num relations, num objects, num objects
        is_left_of_matrix = torch.ones((self.max_nobj, self.max_nobj)) * -1
        is_right_of_matrix = torch.ones((self.max_nobj, self.max_nobj)) * -1
        is_front_of_matrix = torch.ones((self.max_nobj, self.max_nobj)) * -1
        is_behind_of_matrix = torch.ones((self.max_nobj, self.max_nobj)) * -1

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
            obj1 = i + 1
            for j in range(self.max_nobj):
                obj2 = j + 1
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
        return relation_matrix
    


    def __getitem__(self, idx):
        """
        Get the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image tensor, object patches, relations tensor, and mask tensor.
        """
        # Get object patches.
        img, ids, xyz = self._get_data(idx)
        
        obj_patches = self._get_obj_patches()
        
        # Get spatial relations.
        relations_matrix = self.get_spatial_relations(ids, xyz)
       
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

        if self.resize:
            image = F.resize(image, self.img_size, antialias=True)

        return image, obj_patches, relations.float(), mask.reshape(-1).float()

    def __len__(self):
            """
            Returns the length of the dataset.

            Returns:
                int: The length of the dataset.
            """
            return len(self.dataset)

    def _get_obj_patches(self):
        """
        Generate object patches from the image data.

        Returns:
            obj_patches (torch.Tensor): Tensor containing the object patches.
        """
        obj_patches = []

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
        """
        Creates a mask for valid relations based on the given relations matrix.

        Args:
            relations_matrix (numpy.ndarray): The relations matrix containing relation values.

        Returns:
            numpy.ndarray: A mask where '1' indicates a valid relation and '0' indicates no relation.
        """
        mask = relations_matrix != -1

        return mask

    def generate_canidate_patches(self, samples_per_class=12):
        """
        Generates candidate patches for each class in the dataset.

        Args:
            samples_per_class (int): The number of candidate patches to generate per class.

        Returns:
            None
        """

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