import torch
import torch.utils.data as data
import numpy as np
import os
import sys
import random
import json
from PIL import Image
from torchvision import transforms
from .build import DATASETS


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils.io_module import IO
from utils.logger import *



class BasePCNDataset(data.Dataset):
    def __init__(self, config, include_images=False, 
                 num_imgs_per_obj=1,
                 include_segmentation=False,
                 SEG_LIST=None,
                 CLASSES=None):        

        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS
        self.include_images = include_images
        self.num_imgs_per_obj = num_imgs_per_obj
        self.include_segmentation = include_segmentation

        if self.include_images:
            self.img_path = config.IMG_PATH
            self.img_idxs = list(range(24))
            self.img_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor()
            ])
            
        if self.include_segmentation:
            self.SEG_LIST = SEG_LIST
            self.CLASSES = CLASSES

        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        raise NotImplementedError

    def _get_file_list(self, subset, n_renderings=1):
        file_list = []
        for dc in self.dataset_categories:
            samples = dc[subset]
            for s in samples:
                file_list_details = {
                    'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_path': [self.partial_points_path % (subset, dc['taxonomy_id'], s, i) for i in range(n_renderings)],
                    'gt_path': self.complete_points_path % (subset, dc['taxonomy_id'], s, s),
                }
                if self.include_images:
                    img_sample = random.sample(self.img_idxs, self.num_imgs_per_obj)
                    for img_idx in img_sample:
                        file_list_details['img_path'] = self.img_path % (subset, dc['taxonomy_id'], s, img_idx)
                if self.include_segmentation:
                    cls_vec = np.zeros(16)
                    if dc['taxonomy_id'] in self.SEG_LIST:
                        cls_idx = self.CLASSES[dc['taxonomy_id']]
                        cls_vec[cls_idx] = 1
                    file_list_details['cls_vec'] = cls_vec
                file_list.append(file_list_details)
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset == 'train' else 0

        for ri in ['partial', 'gt']:
            file_path = sample[f'{ri}_path']
            if isinstance(file_path, list):
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        if self.include_images:
            img = Image.open(sample['img_path']).convert('RGB')
            img = self.img_transform(img)
            data_img = (data['partial'], data['gt'], img)
        else:
            data_img = (data['partial'], data['gt'])

        # Conditional inclusion of cls_vec
        if 'cls_vec' in sample:
            cls_vec = torch.from_numpy(sample['cls_vec']).float()
            return_data = (*data_img, cls_vec)
        else:
            return_data = data_img

        return sample['taxonomy_id'], sample['model_id'], return_data



    def __len__(self):
        return len(self.file_list)