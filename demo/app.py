import gradio as gr
import os
import sys
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.io_module import IO

current_point_cloud = None


def visualize_point_cloud(file_path, uploaded_file):
    global current_point_cloud
    # Determine the source of the input
    if uploaded_file is not None:
        # Handle file uploaded from the user's computer
        current_point_cloud = IO.get(uploaded_file)
    else:
        # Handle file selected from the dropdown
        current_point_cloud = IO.get(file_path)
    return create_plot(current_point_cloud)


def create_plot(point_cloud):
    if point_cloud is not None:
        fig = go.Figure(data=[go.Scatter3d(
            x=point_cloud[:, 0], 
            y=point_cloud[:, 1], 
            z=point_cloud[:, 2],
            mode='markers',
            marker=dict(size=2)
        )])
        return fig
    else:
        return None


def reconstruct_point_cloud():
    # Apply your reconstruction algorithm on current_point_cloud
    return create_plot(current_point_cloud)


# List of files in the 'samples' directory
samples_dir = os.path.join(BASE_DIR, 'demo', 'samples')
sample_files = ['Select from samples...'] + [f'samples/{file}' for file in os.listdir(samples_dir) if file.endswith('.ply')]

with gr.Blocks() as app:
    gr.Markdown("## 3D Point Cloud Visualization and Reconstruction")
    with gr.Row():
        file_dropdown = gr.Dropdown(label="Select a Sample Point Cloud File", choices=sample_files, value='Select from samples...')
        file_uploader = gr.File(label="Or Upload Your File")
        upload_btn = gr.Button("Load and Visualize")
    original_pc_display = gr.Plot()
    reconstruct_btn = gr.Button("Reconstruct")
    reconstructed_pc_display = gr.Plot()

    upload_btn.click(visualize_point_cloud, inputs=[file_dropdown, file_uploader], outputs=original_pc_display)
    reconstruct_btn.click(reconstruct_point_cloud, inputs=[], outputs=reconstructed_pc_display)

app.launch(share=True)
