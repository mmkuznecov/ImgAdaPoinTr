import torch
import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *
from PIL import Image
from torchvision import transforms


# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class SegImgPCN(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config, num_imgs_per_obj=6):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.img_path = config.IMG_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS
        self.img_idxs = list(range(24))
        self.num_imgs_per_obj = num_imgs_per_obj

        # Load the dataset indexing file
        self.dataset_categories = []
        print('self.category_file  ' * 5, self.category_file) 
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            
        self.seg_list = ['02691156', '03001627', '03636649', '04379243']
        # added for segmentation task
        root = '/home/jovyan/vchopuryan/GDANet/data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
        self.catfile = os.path.join(root, 'synsetoffset2category.txt')
        self.classes = {'02691156': 0,  #Airplane
                        '02773838': 1,  #Bag
                        '02954340': 2,  #Cap
                        '02958343': 3,  #Car
                        '03001627': 4,  #Chair
                        '03261776': 5,  #Earphone
                        '03467517': 6,  #Guitar
                        '03624134': 7,  #knife
                        '03636649': 8,  #Lamp
                        '03642806': 9,  #Laptop
                        '03790512': 10, #Motorbike
                        '03797390': 11, #Mug
                        '03948459': 12, #Pistol
                        '04099429': 13, #Rocket
                        '04225987': 14, #Skateboard
                        '04379243': 15  #Table
                       }
        
        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)
        self.img_transform = transforms.Compose([
            transforms.Scale(224),
            transforms.ToTensor()
        ])
        

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
            samples = dc[subset]
#             print(samples)
            for s in samples:
                if s not in ['c70c1a6a0e795669f51f77a6d7299806',
                             'a8c0ceb67971d0961b17743c18fb63dc',
                             'f3c0ab68f3dab6071b17743c18fb63dc',
                             '2ae70fbab330779e3bff5a09107428a5',
                             '191c92adeef9964c14038d588fd1342f']:
#                 print("(subset, dc['taxonomy_id'], s, i) " * 10, (subset, dc['taxonomy_id'], s, n_renderings))
                    # for segmentation
                    if dc['taxonomy_id'] in self.seg_list:
                        cls_idx = self.classes[dc['taxonomy_id']]
                        cls_idx = np.array([cls_idx]).astype(np.int32)
                        cls_vec = np.zeros(16)
                        cls_vec[cls_idx] = 1
                    else:
                        cls_vec = np.zeros(16)
                        
                    img_sample = random.sample(self.img_idxs, self.num_imgs_per_obj)
                    for img_idx in img_sample:
                        file_list.append({
                            'taxonomy_id':
                            dc['taxonomy_id'],
                            'model_id':
                            s,
                            'partial_path': [
                                self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                                for i in range(n_renderings)
                            ],
                            'gt_path':
                            self.complete_points_path % (subset, dc['taxonomy_id'], s, s),
                            'img_path':
                            self.img_path % (subset, dc['taxonomy_id'], s, img_idx),
                            'cls_vec': cls_vec,
                        })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0
        
        # load view 
        views = self.img_transform(Image.open(sample['img_path']))
        views = views[:3,:,:]

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
#             print('file_path '*50, file_path)
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

#         assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['partial'],
                                                           data['gt'],
                                                           views.float(),
                                                           torch.from_numpy(sample['cls_vec']).float(),
                                                          )

    def __len__(self):
        return len(self.file_list)
    