import numpy as np
import torch
from sklearn.cluster import KMeans
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

def lidar_to_histogram_features(lidar, x_range, y_range):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = x_range // 2
        y_meters_max = y_range // 2
        # here the bev pc is ego-centered, unlike which that in Transfuser is front-viewed
        xbins = np.linspace(-x_meters_max, x_meters_max, 32*pixels_per_meter+1)
        ybins = np.linspace(-y_meters_max, y_meters_max, 32*pixels_per_meter+1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[...,2]<=-2.3]
    above = lidar[lidar[...,2]>-2.3]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([above_features, below_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    features = np.rot90(features, -1, axes=(1,2)).copy()
    return features

def generate_front_view(img):
    '''
    generate front view of 132 FOV from images

    :param img: (BS, 3, H, W, C)
    '''
    width = img.shape[3]
    if not isinstance(img, torch.Tensor):
        img = torch.Tensor(img)
    front_view = img[:,:1,:,:,:].squeeze()
    left_view = img[:,1:2,:,width//2:,:].squeeze()
    right_view = img[:,2:,:,:width//2+1,:].squeeze()
    return torch.cat([left_view, front_view, right_view], dim=2)

def generate_vocabulary(trajectories: Trajectory, voc_size: int=4096) -> np.array:
    '''
    Construct vocabulary size for planning module.

    :param trajectories: The backup trajectories for clustering, [Total num, num states, 3] of (x, y, heading)
    '''
    trajectory_data = trajectories.data # TODO: maybe need to transfer to cpu and numpy array
    kmeans = KMeans(n_clusters=voc_size, max_iter=800).fit(trajectory_data)
    traj_voc = kmeans.cluster_centers_
    return traj_voc