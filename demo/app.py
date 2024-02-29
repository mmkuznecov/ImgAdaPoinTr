import gradio as gr
import os
import sys
import plotly.graph_objects as go
from PIL import Image
import torch
from torchvision import transforms
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.io_module import IO
from pcreconstructor import PCReconstructor, CLASSES

current_class_id = None

class_mapping_path = os.path.join(BASE_DIR, 'cfgs', 'shapenet_synset_dict.json')
with open(class_mapping_path, 'r') as f:
    class_mapping = json.load(f)

def load_and_preprocess_image(image_path, height=224, width=224):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def visualize_point_cloud(label):
    global current_class_id
    # Extract class_id from the label
    class_id = label.split(" - ")[0]
    current_class_id = class_id
    point_cloud_path = os.path.join(BASE_DIR, 'demo', 'samples', class_id, '00.pcd')
    point_cloud = IO.get(point_cloud_path)
    return create_plot(point_cloud)

def create_plot(point_cloud):
    if point_cloud is not None:
        fig = go.Figure(data=[go.Scatter3d(
            x=point_cloud[:, 0], 
            y=point_cloud[:, 1], 
            z=point_cloud[:, 2],
            mode='markers',
            marker=dict(size=2)
        )])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        return fig
    else:
        return None

def reconstruct_point_cloud(model_name):
    global current_class_id
    point_cloud_path = os.path.join(BASE_DIR, 'demo', 'samples', current_class_id, '00.pcd')
    image_path = os.path.join(BASE_DIR, 'demo', 'samples', current_class_id, '00.png')

    # Update predictor based on selected model
    model_configs = {
        'SegEncAdaPoinTr': ('cfgs/SegImgPCN_models/SegEncAdaPoinTr.yaml', 'pretrained/SegEncAdaPoinTr.pth'),
        'ImgResNetEncAdaPoinTrVariableLoss': ('cfgs/ImgPCN_models/ImgResNetEncAdaPoinTrVariableLoss.yaml', 'pretrained/ImgAdaPoinTr.pth'),
        'ImgEncSegDecAdaPoinTrVariableLoss': ('cfgs/SegImgPCN_models/ImgEncSegDecAdaPoinTrVariableLoss.yaml', 'pretrained/ImgEncSegDecAPTr.pth')
    }
    
    use_segmentation = False if model_name == 'ImgResNetEncAdaPoinTrVariableLoss' else True
    
    config_path, ckpt_path = model_configs[model_name]
    predictor = PCReconstructor(config_path, ckpt_path, use_segmentation=use_segmentation)
    
    reconstructed_points = predictor.predict(point_cloud_path, image_path, current_class_id, CLASSES)
    return create_plot(reconstructed_points)

samples_dir = os.path.join(BASE_DIR, 'demo', 'samples')

sample_files = []
for class_id in os.listdir(samples_dir):
    if os.path.isdir(os.path.join(samples_dir, class_id)):
        class_name = class_mapping.get(class_id, "Unknown")
        label = f"{class_id} - {class_name}"
        sample_files.append(label)

# Gradio app setup
with gr.Blocks() as app:
    gr.Markdown("## 3D Point Cloud Visualization and Reconstruction")
    with gr.Row():
        file_dropdown = gr.Dropdown(label="Select an Incomplete Sample Point Cloud File", choices=sample_files, value=sample_files[0])
        model_selection = gr.Dropdown(label="Select a Model", choices=[
            'SegEncAdaPoinTr', 'ImgResNetEncAdaPoinTrVariableLoss', 'ImgEncSegDecAdaPoinTrVariableLoss'
        ], value='SegEncAdaPoinTr')
        visualize_btn = gr.Button("Load and Visualize")
    original_pc_display = gr.Plot()
    reconstruct_btn = gr.Button("Reconstruct")
    reconstructed_pc_display = gr.Plot()

    visualize_btn.click(visualize_point_cloud, inputs=[file_dropdown], outputs=original_pc_display)
    reconstruct_btn.click(reconstruct_point_cloud, inputs=[model_selection], outputs=reconstructed_pc_display)

app.queue()
app.launch(share=True, server_port=8080)