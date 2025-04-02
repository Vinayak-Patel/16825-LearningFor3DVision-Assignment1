import torch
from pytorch3d.structures import Pointclouds
import pytorch3d.renderer as rdr
import imageio

from starter.utils import unproject_depth_image, get_points_renderer
from starter.render_generic import load_rgbd_data
def create_point_cloud(rgb, depth, mask, camera):
    points, colors = unproject_depth_image(
        rgb,
        depth,
        mask,
        camera
    )
    
    point_cloud = Pointclouds(
        points=[points],
        features=[colors]
    )
    
    return point_cloud

def render_point_clouds(point_cloud, device, num_views=30):
    renderer = get_points_renderer(
        image_size=512,
        device=device
    )
    
    images = []
    angles = torch.linspace(-180, 180, num_views)
    
    for angle in angles:
        R, T = rdr.look_at_view_transform(
            dist=6.0,
            elev=30.0,
            azim=angle.item()
        )
        cameras = rdr.FoVOrthographicCameras(
            R=R,
            T=T,
            device=device
        )
        
        rend = renderer(point_cloud, cameras=cameras)
        image = rend[0, ..., :3].cpu().numpy()
        images.append(image)
    
    return images

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dict = load_rgbd_data()
print("Available keys in data_dict:", data_dict.keys())

pc1 = create_point_cloud(
    data_dict["rgb1"],
    data_dict["depth1"],
    data_dict["mask1"],
    data_dict["camera1"]
)

pc2 = create_point_cloud(
    data_dict["rgb2"],
    data_dict["depth2"],
    data_dict["mask2"],
    data_dict["camera2"]
)

combined_points = torch.cat([pc1.points_list()[0], pc2.points_list()[0]], dim=0)
combined_colors = torch.cat([pc1.features_list()[0], pc2.features_list()[0]], dim=0)
pc_combined = Pointclouds(points=[combined_points], features=[combined_colors])

renders1 = render_point_clouds(pc1, device)
renders2 = render_point_clouds(pc2, device)
renders_combined = render_point_clouds(pc_combined, device)

imageio.mimsave('pointcloud1.gif', renders1, fps=15)
imageio.mimsave('pointcloud2.gif', renders2, fps=15)
imageio.mimsave('pointcloud_combined.gif', renders_combined, fps=15)