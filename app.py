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
from src.utils.mesh_util import save_obj, save_glb, save_stl
from src.utils.infer_util import remove_background, resize_foreground, images_to_video

import tempfile
from huggingface_hub import hf_hub_download
import datetime


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

device = torch.device('cuda')

# load diffusion model
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
    cache_dir=model_cache_dir
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet
unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model", cache_dir=model_cache_dir)
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)



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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    show_image = np.asarray(z123_image, dtype=np.uint8)
    show_image_copy = show_image.copy()
    show_image = torch.from_numpy(show_image_copy)     # (960, 640, 3)
    show_image = rearrange(show_image, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_image = rearrange(show_image, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    show_image = Image.fromarray(show_image.numpy())

    return z123_image, show_image


def make_mesh(mesh_fpath, planes):

    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    mesh_glb_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.glb")
    mesh_stl_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.stl")
        
    with torch.no_grad():
        # get mesh

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=False,
            **infer_config,
        )

        vertices, faces, vertex_colors = mesh_out
        vertices = vertices[:, [1, 2, 0]]
        
        save_glb(vertices, faces, vertex_colors, mesh_glb_fpath)
        save_stl(vertices, faces, mesh_stl_fpath)
        save_obj(vertices, faces, vertex_colors, mesh_fpath)
        
        print(f"Mesh saved to {mesh_fpath}")

    return mesh_fpath, mesh_glb_fpath, mesh_stl_fpath


def make3d(images):
    model.to(device1)

    images = np.asarray(images, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device1)
    render_cameras = get_render_cameras(
        batch_size=1, radius=4.5, elevation=20.0, is_flexicubes=IS_FLEXICUBES).to(device1)

    images = images.unsqueeze(0).to(device1)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    mesh_fpath = tempfile.NamedTemporaryFile(suffix=f".obj", delete=False).name
    print(mesh_fpath)
    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    video_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.mp4")

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

    mesh_fpath, mesh_glb_fpath, mesh_stl_fpath = make_mesh(mesh_fpath, planes)

    model.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return video_fpath, mesh_fpath, mesh_glb_fpath, mesh_stl_fpath


import gradio as gr


def load_history():
    history_dir = "history"
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
        return []
    
    try:
        history_files = [os.path.join(history_dir, f) for f in os.listdir(history_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        history_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return history_files
    except OSError:
        return []



def load_history():
    history_dir = "history"
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
        return []
    
    try:
        history_files = [os.path.join(history_dir, f) for f in os.listdir(history_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        history_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
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
                    sources="upload",
                    width=256,
                    height=256,
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(
                    label="Processed Image", 
                    image_mode="RGBA", 
                    width=256,
                    height=256,
                    type="pil", 
                    interactive=False
                )
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    sample_seed = gr.Number(value=42, label="Seed Value", precision=0)

                    sample_steps = gr.Slider(
                        label="Sample Steps",
                        minimum=30,
                        maximum=75,
                        value=75,
                        step=5
                    )

            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")

            with gr.Accordion("Examples", open=False):
                gr.Examples(
                    examples=[
                        os.path.join("examples", img_name) for img_name in sorted(os.listdir("examples"))
                    ],
                    inputs=[input_image],
                    examples_per_page=20
                )

            with gr.Accordion("History", open=True):
                history_gallery = gr.Gallery(
                    label="Generation History",
                    show_label=False,
                    elem_id="history_gallery",
                    columns=4,
                    height="auto",
                    object_fit="contain",
                )
                delete_button = gr.Button("Delete Selected Image")

        with gr.Column(scale=2):

            with gr.Row():

                with gr.Column():
                    mv_show_images = gr.Image(
                        label="Generated Multi-views",
                        type="pil",
                        width=379,
                        interactive=False
                    )

                with gr.Column():
                    output_video = gr.Video(
                        label="video", format="mp4",
                        width=379,
                        autoplay=True,
                        interactive=False
                    )

            with gr.Row():
                with gr.Tab("OBJ"):
                    output_model_obj = gr.Model3D(
                        label="Output Model (OBJ Format)",
                        #width=768,
                        interactive=False,
                    )
                    gr.Markdown("Note: Downloaded .obj model will be flipped. Export .glb instead or manually flip it before usage.")
                with gr.Tab("GLB"):
                    output_model_glb = gr.Model3D(
                        label="Output Model (GLB Format)",
                        #width=768,
                        interactive=False,
                    )
                    gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")
                with gr.Tab("STL"):
                    output_model_stl = gr.Model3D(
                        label="Output Model (STL Format)",
                        #width=768,
                        interactive=False,
                    )

            with gr.Row():
                gr.Markdown('''Try a different <b>seed value</b> if the result is unsatisfying (Default: 42).''')

    gr.Markdown(_CITE_)
    mv_images = gr.State()
    image_history = gr.State(value=load_history())
    selected_index = gr.State(None)



    def add_to_history(image, history):
        temp_dir = "history"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        filename = os.path.join(temp_dir, f"{timestamp}.png")
        if image is not None:
            try:
                image.save(filename)
                history.insert(0, filename) # Add to the beginning
            except Exception as e:
                print(f"Failed to save history image: {e}")
        
        return history, gr.update(value=history)

    def on_history_select(evt: gr.SelectData):
        if evt.value:
            try:
                return Image.open(evt.value).convert("RGBA"), evt.index
            except Exception as e:
                print(f"Error loading history image: {e}")
        return gr.update(), gr.update()

    def delete_image(history, index):
        if index is not None and 0 <= index < len(history):
            filepath = history.pop(index)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Failed to delete history image: {e}")
        return history, gr.update(value=history), None

    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background],
        outputs=[processed_image],
    ).success(
        fn=generate_mvs,
        inputs=[processed_image, sample_steps, sample_seed],
        outputs=[mv_images, mv_show_images],
    ).success(
        fn=make3d,
        inputs=[mv_images],
        outputs=[output_video, output_model_obj, output_model_glb, output_model_stl]
    ).success(
        fn=add_to_history,
        inputs=[processed_image, image_history],
        outputs=[image_history, history_gallery]
    )

    history_gallery.select(
        fn=on_history_select,
        outputs=[input_image, selected_index],
        show_progress=False
    )
    delete_button.click(
        fn=delete_image,
        inputs=[image_history, selected_index],
        outputs=[image_history, history_gallery, selected_index],
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
