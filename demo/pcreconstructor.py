import torch
import os
import sys
import json
from torchvision import transforms
from PIL import Image


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.config import get_config
from tools.builder import model_builder, load_model
from utils.io_module import IO

def load_and_preprocess_image(image_path, height=224, width=224):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

class PCReconstructor:
    def __init__(self, config_path, ckpt_path, use_imgs=True, use_segmentation=True):
        self.use_imgs = use_imgs
        self.use_segmentation = use_segmentation
        args = type('', (), {})()
        args.config = config_path
        args.ckpts = ckpt_path
        args.use_gpu = torch.cuda.is_available()
        args.local_rank = 0
        args.resume = False
        args.distributed = False
        args.launcher = 'none'
        args.experiment_path = './experiments'
        args.tfboard_path = './tf_logs'

        os.makedirs(args.experiment_path, exist_ok=True)
        os.makedirs(args.tfboard_path, exist_ok=True)

        config = get_config(args)
        model = model_builder(config.model)
        load_model(model, args.ckpts, logger=None)
        
        device = 'cuda' if args.use_gpu else 'cpu'
        self.device = device
        
        model = model.to(device)
        model.eval()
        self.model = model
        
    def predict(self, point_cloud_path, image_path, class_id, classes):
        xyz = self.preprocess_point_cloud(point_cloud_path)
        inputs = [xyz]  # Start with point cloud input
        
        if self.use_imgs:
            img = load_and_preprocess_image(image_path)
            img = img.to(self.device)
            inputs.append(img)  # Add image input if required
        
        if self.use_segmentation:
            cls_vec = self.prepare_cls_vec(classes, class_id)
            cls_vec = cls_vec.to(self.device)
            inputs.append(cls_vec)  # Add class vector input if required
        
        with torch.no_grad():
            output = self.model(*inputs)
            reconstructed_points = output[-1].squeeze().cpu().numpy()
        return reconstructed_points

    
    def preprocess_point_cloud(self, point_cloud_path, num_points=2048):
        point_cloud = IO.get(point_cloud_path)
        point_cloud = point_cloud[:num_points]
        point_cloud = torch.tensor(point_cloud, dtype=torch.float).unsqueeze(0)  # Add batch dimension
        if torch.cuda.is_available():
            point_cloud = point_cloud.cuda()
        return point_cloud

    def prepare_cls_vec(self, classes, class_id):
        num_classes = len(classes)
        cls_vec = torch.zeros(num_classes)
        class_index = classes.get(class_id, -1)
        if class_index != -1:
            cls_vec[class_index] = 1
        return cls_vec.unsqueeze(0)  # Add batch dimension