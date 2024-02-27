from .build import DATASETS
from .BasePCNDataset import BasePCNDataset
from . import data_transforms


@DATASETS.register_module()
class PCN(BasePCNDataset):
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


@DATASETS.register_module()
class PCNv2(BasePCNDataset):
    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([
                {'callback': 'UpSamplePoints', 'parameters': {'n_points': 2048}, 'objects': ['partial']},
                {'callback': 'RandomMirrorPoints', 'objects': ['partial', 'gt']},
                {'callback': 'ToTensor', 'objects': ['partial', 'gt']}
            ])
        else:
            return data_transforms.Compose([
                {'callback': 'UpSamplePoints', 'parameters': {'n_points': 2048}, 'objects': ['partial']},
                {'callback': 'ToTensor', 'objects': ['partial', 'gt']}
            ])
