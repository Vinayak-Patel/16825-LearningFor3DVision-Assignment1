import torch
import pytorch3d
import imageio
import numpy as np
from tqdm import tqdm

from utils import get_mesh_renderer
from render_mesh import render_cow
def render_360_degree_mesh(
    mesh,
    device,
    image_size=512,
    num_views=72,
    distance=2.7,
    elevation=30
):
    
    renderer = get_mesh_renderer(image_size=image_size)
    angles = torch.linspace(-180,180,num_views)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -4]], device=device)
    
    images = []
    
    for angle in tqdm(angles, desc="Views Rendered"):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=distance,
            elev=elevation,
            azim=angle
        )
    
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        
        render = renderer(mesh, cameras, lights)
        
        image = render[0,...,:3].cpu().numpy()
        
        images.append(image)

    return images    

def save_gif(images, output_path, fps=24):
    duration = 1000/24
    
    imageio.mimsave(output_path, images, duration=duration, loop=0)
    
device = torch.device('cuda')

obj_loc = "../data/cow.obj"
mesh = render_cow(cow_path="data/cow.obj", image_size=512, color=[0.7, 0.7, 1], device=None,)
# load_objs_as_meshes([obj_loc], device=device)[3][4]
render_img = render_360_degree_mesh(
    mesh,
    device=device,
    image_size=512,
    num_views=72,
    distance=2.8,
    elevation=30
)

save_gif(render_img, 'cow_turntable.gif', fps=24)