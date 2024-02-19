from .build import DATASETS
from .BasePCNDataset import BasePCNDataset
import data_transforms

SEG_LIST = ['02691156', '03001627', '03636649', '04379243']

CLASSES = {
    '02691156': 0, '02773838': 1, '02954340': 2, '02958343': 3, '03001627': 4,
    '03261776': 5, '03467517': 6, '03624134': 7, '03636649': 8, '03642806': 9,
    '03790512': 10, '03797390': 11, '03948459': 12, '04099429': 13,
    '04225987': 14, '04379243': 15
}


@DATASETS.register_module()
class SegImgPCN(BasePCNDataset):
    def __init__(self, config, num_imgs_per_obj=6):
        super().__init__(config, include_images=True, 
                         num_imgs_per_obj=num_imgs_per_obj, 
                         include_segmentation=True,
                         SEG_LIST=SEG_LIST,
                         CLASSES=CLASSES)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([
                {'callback': 'RandomSamplePoints', 'parameters': {'n_points': 2048}, 'objects': ['partial']},
                {'callback': 'RandomMirrorPoints', 'objects': ['partial', 'gt']},
                {'callback': 'ToTensor', 'objects': ['partial', 'gt']}
            ])
        else:
            return data_transforms.Compose([
                {'callback': 'RandomSamplePoints', 'parameters': {'n_points': 2048}, 'objects': ['partial']},
                {'callback': 'ToTensor', 'objects': ['partial', 'gt']}
            ])
