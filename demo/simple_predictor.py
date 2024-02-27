from models.internal.SegEncAdaPoinTr_backup import SegEncAdaPoinTr
from datasets.data_transforms import *
from datasets.io_module import IO
from utils.config import get_config
from tools.builder import model_builder, load_model
import os
import json

def load_class_mapping(json_path):
    with open(json_path, 'r') as file:
        class_mapping = json.load(file)
    return class_mapping


def load_pretrained_model(config_path, ckpt_path):
    # Create an empty args object with necessary attributes
    args = type('', (), {})()
    args.config = config_path
    args.ckpts = ckpt_path
    args.use_gpu = torch.cuda.is_available()
    args.local_rank = 0  # Assuming using a single GPU or CPU
    args.resume = False  # Assuming we're not resuming training
    args.distributed = False  # Assuming not using distributed training
    args.launcher = 'none'  # No launcher used
    args.experiment_path = './experiments'  # Default experiment path
    args.tfboard_path = './tf_logs'  # Default path for tensorboard logs

    # Ensure the experiment and tensorboard log paths exist
    os.makedirs(args.experiment_path, exist_ok=True)
    os.makedirs(args.tfboard_path, exist_ok=True)

    # Load the configuration
    config = get_config(args)
    model = model_builder(config.model)
    
    # Load pretrained weights
    load_model(model, args.ckpts, logger=None)  # Assuming logger is not required here
    
    if args.use_gpu:
        model.cuda()
    model.eval()
    return model

# Function to preprocess input point cloud
def preprocess_point_cloud(file_path, num_points=2048):
    point_cloud = IO.get(file_path)
    # Assuming the point cloud is not already sampled to the desired number of points
    # You might need to implement or adjust this function based on your needs
    point_cloud = point_cloud[:num_points]
    point_cloud = torch.tensor(point_cloud, dtype=torch.float).unsqueeze(0)  # Add batch dimension
    if torch.cuda.is_available():
        point_cloud = point_cloud.cuda()
    return point_cloud


def predict(model, xyz, img, cls_vec):
    with torch.no_grad():
        output = model(xyz, img, cls_vec)
        # Assuming the output is the last layer output
        reconstructed_points = output[-1].squeeze().cpu().numpy()
    return reconstructed_points


def prepare_cls_vec(classes, class_id):
    num_classes = len(classes)
    cls_vec = torch.zeros(num_classes)
    class_index = classes[class_id]
    cls_vec[class_index] = 1
    return cls_vec.unsqueeze(0)

def prepare_dummy_img(height=224, width=224, channels=3):
    return torch.rand(1, channels, height, width)  # Create a dummy image tensor


if __name__ == "__main__":
    config_path = 'cfgs/SegImgPCN_models/SegEncAdaPoinTr.yaml'
    ckpt_path = 'pretrained/SegEncAdaPoinTr.pth'
    model = load_pretrained_model(config_path, ckpt_path)
    
    example_ply_path = '/home/jovyan/ImgAdaPoinTr/demo/samples/plane.ply'
    input_tensor = preprocess_point_cloud(example_ply_path)
    
    CLASSES = {
        '02691156': 0, '02773838': 1, '02954340': 2, '02958343': 3, '03001627': 4,
        '03261776': 5, '03467517': 6, '03624134': 7, '03636649': 8, '03642806': 9,
        '03790512': 10, '03797390': 11, '03948459': 12, '04099429': 13,
        '04225987': 14, '04379243': 15
    }
    
    cls_vec = prepare_cls_vec(CLASSES, "02691156")  # Class ID for "airplane"
    dummy_img = prepare_dummy_img()
    
    if torch.cuda.is_available():
        cls_vec = cls_vec.cuda()
        dummy_img = dummy_img.cuda()
    
    reconstructed_points = predict(model, input_tensor, dummy_img, cls_vec)
    print(reconstructed_points.shape)