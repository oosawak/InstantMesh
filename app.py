import os
import imageio
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
# from pytorch_lightning import seed_everything
from lightning.pytorch  import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_glb, save_stl, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, images_to_video

import tempfile
from huggingface_hub import hf_hub_download
import datetime
import gc
import shutil
import shutil
import glob


if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
else:
    device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device1 = device0

# Define the cache directory for model files
model_cache_dir = './ckpts/'
os.makedirs(model_cache_dir, exist_ok=True)

def get_render_cameras(batch_size=1, M=120, radius=2.5, elevation=10.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def images_to_video(images, output_path, fps=30):
    # images: (N, C, H, W)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
        assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
            f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert frame.min() >= 0 and frame.max() <= 255, \
            f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_path, np.stack(frames), fps=fps, codec='h264')


###############################################################################
# Configuration.
###############################################################################

seed_everything(0)

config_path = 'configs/instant-mesh-large.yaml'
config = OmegaConf.load(config_path)
config_name = os.path.basename(config_path).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

# load diffusion model
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    cache_dir=model_cache_dir
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet
unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model", cache_dir=model_cache_dir)
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)
pipeline.to('cpu')


# load reconstruction model
print('Loading reconstruction model ...')
model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_large.ckpt", repo_type="model", cache_dir=model_cache_dir)
model = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
model.load_state_dict(state_dict, strict=True)

model = model.to(device1)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device1, fovy=30.0)
model = model.eval()
model.to('cpu')

print('Loading Finished!')


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background):

    rembg_session = rembg.new_session() if do_remove_background else None
    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

    return input_image


def generate_mvs(input_image, sample_steps, sample_seed):

    seed_everything(sample_seed)
    
    pipeline.to(device0)
    # sampling
    generator = torch.Generator(device=device0)
    z123_image = pipeline(
        input_image, 
        num_inference_steps=sample_steps, 
        generator=generator,
    ).images[0]
    
    pipeline.to('cpu')
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    show_image = np.asarray(z123_image, dtype=np.uint8)
    show_image_copy = show_image.copy()
    show_image = torch.from_numpy(show_image_copy)     # (960, 640, 3)
    show_image = rearrange(show_image, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_image = rearrange(show_image, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    show_image = Image.fromarray(show_image.numpy())

    return z123_image, show_image


def make_mesh(mesh_fpath, planes, export_texmap):

    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    
    with torch.no_grad():
        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=export_texmap,
            **infer_config,
        )

        if export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            
            # Apply the same coordinate transformation as the other branch
            vertices_transformed = vertices[:, [1, 2, 0]]

            # Save OBJ with texture map using transformed vertices
            save_obj_with_mtl(
                vertices_transformed.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_fpath,
            )
            print(f"Mesh with texture map saved to {mesh_fpath}")

                        # Also save GLB and STL versions (without texture information)
            mesh_glb_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.glb")
            mesh_stl_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.stl")
            
            save_glb(vertices_transformed.cpu(), faces.cpu(), None, mesh_glb_fpath)
            save_stl(vertices_transformed.cpu(), faces.cpu(), mesh_stl_fpath)
            
            return mesh_fpath, mesh_glb_fpath, mesh_stl_fpath
        else:
            mesh_glb_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.glb")
            mesh_stl_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.stl")
            
            vertices, faces, vertex_colors = mesh_out
            vertices = vertices[:, [1, 2, 0]]
            
            save_glb(vertices, faces, vertex_colors, mesh_glb_fpath)
            save_stl(vertices, faces, mesh_stl_fpath)
            save_obj(vertices, faces, vertex_colors, mesh_fpath)
            
            print(f"Mesh saved to {mesh_fpath}")
            return mesh_fpath, mesh_glb_fpath, mesh_stl_fpath


def make3d(images, export_texmap):
    model.to(device1)

    images = np.asarray(images, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device1)
    render_cameras = get_render_cameras(
        batch_size=1, radius=4.5, elevation=20.0, is_flexicubes=IS_FLEXICUBES).to(device1)

    images = images.unsqueeze(0).to(device1)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    # Create a temporary directory for all output files
    temp_dir = tempfile.mkdtemp()
    mesh_fpath = os.path.join(temp_dir, "mesh.obj")
    # The video path should also be in the temporary directory
    video_fpath = os.path.join(temp_dir, "video.mp4")

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)

        # get video
        chunk_size = 20 if IS_FLEXICUBES else 1
        render_size = 384
        
        frames = []
        for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
            if IS_FLEXICUBES:
                frame = model.forward_geometry(
                    planes,
                    render_cameras[:, i:i+chunk_size],
                    render_size=render_size,
                )['img']
            else:
                frame = model.synthesizer(
                    planes,
                    cameras=render_cameras[:, i:i+chunk_size],
                    render_size=render_size,
                )['images_rgb']
            frames.append(frame)
        frames = torch.cat(frames, dim=1)

        images_to_video(
            frames[0],
            video_fpath,
            fps=30,
        )

        print(f"Video saved to {video_fpath}")

    mesh_fpath, mesh_glb_fpath, mesh_stl_fpath = make_mesh(mesh_fpath, planes, export_texmap)

    model.to('cpu')
    del images, input_cameras, render_cameras, planes, frames
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Return the temporary directory path as well
    return video_fpath, mesh_fpath, mesh_glb_fpath, mesh_stl_fpath, temp_dir


import gradio as gr


def load_history():
    history_dir = "history"
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
        return []
    
    try:
        # Get all subdirectories in the history folder
        subdirs = [os.path.join(history_dir, d) for d in os.listdir(history_dir) if os.path.isdir(os.path.join(history_dir, d))]
        # Sort by modification time
        subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        # Get the preview.png from each subdirectory
        history_files = [os.path.join(d, 'preview.png') for d in subdirs if os.path.exists(os.path.join(d, 'preview.png'))]
        return history_files
    except OSError:
        return []


_HEADER_ = '''
<!--
<h2><b>Official 🤗 Gradio Demo</b></h2><h2><a href='https://github.com/TencentARC/InstantMesh' target='_blank'><b>InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models</b></a></h2>

**InstantMesh** is a feed-forward framework for efficient 3D mesh generation from a single image based on the LRM/Instant3D architecture.
-->

Code: <a href='https://github.com/TencentARC/InstantMesh' target='_blank'>GitHub</a>. Techenical report: <a href='https://arxiv.org/abs/2404.07191' target='_blank'>ArXiv</a>.

❗️❗️❗️**Important Notes:**
- Our demo can export a .obj mesh with vertex colors or a .glb mesh now. If you prefer to export a .obj mesh with a **texture map**, please refer to our <a href='https://github.com/TencentARC/InstantMesh?tab=readme-ov-file#running-with-command-line' target='_blank'>Github Repo</a>.
- The 3D mesh generation results highly depend on the quality of generated multi-view images. Please try a different **seed value** if the result is unsatisfying (Default: 42).
'''

_CITE_ = r"""
If InstantMesh is helpful, please help to ⭐ the <a href='https://github.com/TencentARC/InstantMesh' target='_blank'>Github Repo</a>. Thanks! [![GitHub Stars](https://img.shields.io/github/stars/TencentARC/InstantMesh?style=social)](https://github.com/TencentARC/InstantMesh)
---
📝 **Citation**

If you find our work useful for your research or applications, please cite using this bibtex:
```bibtex
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2404.07191},
  year={2024}
}
```

📋 **License**

Apache-2.0 LICENSE. Please refer to the [LICENSE file](https://huggingface.co/spaces/TencentARC/InstantMesh/blob/main/LICENSE) for details.

📧 **Contact**

If you have any questions, feel free to open a discussion or contact us at <b>bluestyle928@gmail.com</b>.
"""

with gr.Blocks() as demo:
    gr.Markdown(_HEADER_)
    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources=["upload"],
                    width=256,
                    height=256,
                    type="pil",
                )
                processed_image = gr.Image(
                    label="Processed Image", 
                    image_mode="RGBA", 
                    width=256,
                    height=256,
                    type="pil",
                )
            with gr.Group():
                do_remove_background = gr.Checkbox(
                    label="Remove Background", value=True
                )
                export_texmap = gr.Checkbox(
                    label="Export Texture Map", value=False
                )
                sample_seed = gr.Number(value=42, label="Seed Value", precision=0)
                sample_steps = gr.Slider(
                    label="Sample Steps",
                    minimum=30,
                    maximum=75,
                    value=75,
                    step=5
                )
            submit = gr.Button("Generate", variant="primary")
            with gr.Accordion("Examples", open=False):
                gr.Examples(
                    examples=[
                        os.path.join("examples", img_name) for img_name in sorted(os.listdir("examples"))
                    ],
                    inputs=input_image,
                    examples_per_page=20
                )
            with gr.Accordion("History", open=True):
                delete_button = gr.Button("Delete Selected Image")
                history_gallery = gr.Gallery(
                    label="Generation History",
                    show_label=False,
                    columns=4,
                    height="auto",
                    object_fit="contain",
                )
                with gr.Row(visible=False) as history_download_row:
                    history_obj_download = gr.File(label="Download OBJ", interactive=False)
                    history_glb_download = gr.File(label="Download GLB", interactive=False)
                    history_stl_download = gr.File(label="Download STL", interactive=False)

        with gr.Column(scale=2):
            mv_show_images = gr.Image(
                label="Generated Multi-views",
                type="pil",
                width=379,
            )
            output_video = gr.Video(
                label="video", format="mp4",
                width=379,
                autoplay=True,
            )
            with gr.Tabs():
                with gr.Tab("OBJ"):
                    output_model_obj = gr.Model3D(
                        label="Output Model (OBJ Format)",
                    )
                    gr.Markdown("Note: Downloaded .obj model will be flipped. Export .glb instead or manually flip it before usage.")
                with gr.Tab("GLB"):
                    output_model_glb = gr.Model3D(
                        label="Output Model (GLB Format)",
                    )
                    gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")
                with gr.Tab("STL"):
                    output_model_stl = gr.Model3D(
                        label="Output Model (STL Format)",
                    )
            gr.Markdown('''Try a different <b>seed value</b> if the result is unsatisfying (Default: 42).''')

    gr.Markdown(_CITE_)
    mv_images = gr.State()
    temp_dir_state = gr.State() # To store the temporary directory path
    image_history = gr.State(value=load_history())
    selected_index = gr.State()

    def add_to_history(image, temp_dir, history):
        if not temp_dir or not os.path.isdir(temp_dir):
            print(f"Invalid temporary directory: {temp_dir}")
            return history, gr.update(value=history)

        history_root = "history"
        os.makedirs(history_root, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        new_history_dir = os.path.join(history_root, timestamp)
        os.makedirs(new_history_dir, exist_ok=True)

        # Save the preview image
        preview_path = os.path.join(new_history_dir, "preview.png")
        if image is not None:
            try:
                image.save(preview_path)
                history.insert(0, preview_path) # Add to the beginning
            except Exception as e:
                print(f"Failed to save history image: {e}")
        
        # Move all files from temp_dir to the new history directory
        for filename in os.listdir(temp_dir):
            shutil.move(os.path.join(temp_dir, filename), new_history_dir)
        
        # Clean up the now-empty temporary directory
        try:
            os.rmdir(temp_dir)
        except OSError as e:
            print(f"Error removing temporary directory {temp_dir}: {e}")

        return history, gr.update(value=history)

    def on_history_select(evt: gr.SelectData, history: list):
        if evt.index is None or not history:
            return gr.update(), gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
        try:
            # Use the index to get the correct path from our state, not the temp path from the event value
            image_path = history[evt.index]

            if os.path.exists(image_path):
                # The history item is the preview.png, its parent is the directory
                history_item_dir = os.path.dirname(image_path)
                
                obj_path = glob.glob(os.path.join(history_item_dir, "*.obj"))
                glb_path = glob.glob(os.path.join(history_item_dir, "*.glb"))
                stl_path = glob.glob(os.path.join(history_item_dir, "*.stl"))

                # Make the download row and buttons visible
                return (
                    Image.open(image_path).convert("RGBA"), 
                    evt.index,
                    gr.update(visible=True),
                    gr.update(value=obj_path[0] if obj_path else None, visible=bool(obj_path)),
                    gr.update(value=glb_path[0] if glb_path else None, visible=bool(glb_path)),
                    gr.update(value=stl_path[0] if stl_path else None, visible=bool(stl_path)),
                )
        except Exception as e:
            print(f"Error loading history item: {e}")
        
        # Fallback if something goes wrong
        return gr.update(), gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


    def delete_image(history, index):
        if index is not None and 0 <= index < len(history):
            item_to_remove = history.pop(index)
            # item_to_remove is the path to preview.png
            history_item_dir = os.path.dirname(item_to_remove)
            if os.path.isdir(history_item_dir):
                try:
                    shutil.rmtree(history_item_dir)
                    print(f"Deleted history directory: {history_item_dir}")
                except Exception as e:
                    print(f"Failed to delete history directory: {e}")
        # Hide download buttons after deletion
        return history, gr.update(value=history), None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    submit.click(fn=check_input_image, inputs=[input_image]).then(
        fn=preprocess,
        inputs=[input_image, do_remove_background],
        outputs=[processed_image],
    ).then(
        fn=generate_mvs,
        inputs=[processed_image, sample_steps, sample_seed],
        outputs=[mv_images, mv_show_images],
    ).then(
        fn=make3d,
        inputs=[mv_images, export_texmap],
        outputs=[output_video, output_model_obj, output_model_glb, output_model_stl, temp_dir_state]
    ).then(
        fn=add_to_history,
        inputs=[processed_image, temp_dir_state, image_history],
        outputs=[image_history, history_gallery]
    )

    history_gallery.select(
        fn=on_history_select,
        inputs=[image_history],
        outputs=[
            input_image, 
            selected_index, 
            history_download_row,
            history_obj_download,
            history_glb_download,
            history_stl_download
        ],
        show_progress=False
    )
    delete_button.click(
        fn=delete_image,
        inputs=[image_history, selected_index],
        outputs=[
            image_history, 
            history_gallery, 
            selected_index,
            history_download_row,
            history_obj_download,
            history_glb_download,
            history_stl_download
        ],
    )

    def update_history_gallery(history):
        return gr.update(value=history)

    demo.load(
        fn=lambda: load_history(),
        outputs=[image_history]
    ).then(
        fn=update_history_gallery,
        inputs=[image_history],
        outputs=[history_gallery]
    )


demo.queue(max_size=10)
demo.launch(server_name="0.0.0.0", server_port=43839)
