import torch
import os
import json
from torchvision import transforms
from PIL import Image
from utils.config import get_config
from tools.builder import model_builder, load_model
from datasets.io_module import IO

CLASSES = {
        '02691156': 0, '02773838': 1, '02954340': 2, '02958343': 3, '03001627': 4,
        '03261776': 5, '03467517': 6, '03624134': 7, '03636649': 8, '03642806': 9,
        '03790512': 10, '03797390': 11, '03948459': 12, '04099429': 13,
        '04225987': 14, '04379243': 15
    }

def load_and_preprocess_image(image_path, height=224, width=224):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

class Predictor:
    def __init__(self, config_path, ckpt_path):
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
        
    def predict(self, point_cloud_path, img, class_id, classes):
        xyz = self.preprocess_point_cloud(point_cloud_path)
        cls_vec = self.prepare_cls_vec(classes, class_id)
            
        img = img.to(self.device)
        cls_vec = cls_vec.to(self.device)
        
        with torch.no_grad():
            output = self.model(xyz, img, cls_vec)
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
        if class_index == -1:
            raise ValueError(f"Class ID {class_id} not found in provided classes.")
        cls_vec[class_index] = 1
        return cls_vec.unsqueeze(0)  # Add batch dimension

if __name__ == "__main__":
    root_dir = '/home/jovyan/samples/good_samples'  # Update this path to your root directory
    config_path = 'cfgs/SegImgPCN_models/SegEncAdaPoinTr.yaml'
    ckpt_path = 'pretrained/SegEncAdaPoinTr.pth'
    
    predictor = Predictor(config_path, ckpt_path)
    
    for class_id in os.listdir(root_dir):
        
        try:
        
            class_dir = os.path.join(root_dir, class_id)
            if os.path.isdir(class_dir):
                point_cloud_path = os.path.join(class_dir, "00.pcd")
                image_path = os.path.join(class_dir, "00.png")
                print(class_id)
                img = load_and_preprocess_image(image_path)
                img = img.to(predictor.device)  # Move image tensor to the correct device
                
                reconstructed_points = predictor.predict(point_cloud_path, img, class_id, CLASSES)
                print(f"Class ID: {class_id}, Reconstructed Points Shape: {reconstructed_points.shape}")
                
            print(f"{class_id} NOOOOORM")
        
        except:
            print(f"{class_id} ERROR")