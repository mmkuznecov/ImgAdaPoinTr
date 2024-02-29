from .build import DATASETS
from .BasePCNDataset import BasePCNDataset
from . import data_transforms


@DATASETS.register_module()
class ImgPCN(BasePCNDataset):
    def __init__(self, config, num_imgs_per_obj=1):
        super().__init__(config, include_images=True, num_imgs_per_obj=num_imgs_per_obj)

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
