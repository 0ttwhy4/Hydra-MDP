from __future__ import annotations

from typing import Dict, Optional, Type

from click import Option
import numpy as np
import numpy.typing as npt
import torch

from demo.tools.transfuser_data import lidar_to_histogram_features, generate_front_view

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from feature.imagelidar import ImageLidar
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.simulation.observation.observation_type import (
    CameraChannel,
    LidarChannel,
    SensorChannel,
    Sensors,
)


class ImageLidarFeatureBuilder(AbstractFeatureBuilder):
    """
    Image-lidar mixed feature builder for constructing model input features.
    """
    
    def __init__(self, 
                 backbone, 
                 config: Optional[dict]): # TODO: define a proper initialization (hydra)
        pass
    
    def get_feature_unique_name(cls) -> str:
        return "imagelidar"
    
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        return ImageLidar
    
    def get_features_from_scenario(self, scenario: NuPlanScenario, iteration: int,velocity:Optional[torch.Tensor]):
        channels = [LidarChannel.MERGED_PC, CameraChannel.CAM_F0, CameraChannel.CAM_L1, CameraChannel.CAM_R1]
        sensors = scenario.get_sensors_at_iteration(iteration, channels)
        lidar, imgs = sensors.pointcloud, sensors.images
        lidar, imgs = self.preprocess(imgs, lidar)
        fused_feat = self.forward(imgs, lidar, velocity)
        return ImageLidar(fused_feat)
    
    def get_features_from_simulation(self):
        """
        Get data from nuplan dataset and duplicate the simulation methods
        """
        return NotImplemented
    
    def preprocess(self, img, lidar, img_size, lidar_size, img_transform=None, lidar_transform=None) -> ImageLidar:
        '''
        Perform preprocessing to image and lidar: size, reshaping of front view, transformation to bev.
        Note that the bev is ego-centered.
        You can get this from transfuser/data.py
        
        :param lidar_size: the size [m] of lidar map in real world
        '''
        # TODO: projection to bev space
        lidar = lidar_to_histogram_features(lidar)
        if img_transform is not None:
            img_data = img_transform(img)
        if lidar_transform is not None:
            lidar_data = lidar_transform(lidar)
        assert img.shape.length == 5, "the shape of input image should be (BS, Views, H, W, C)"
        # TODO: may require update according to the nuplan dataset
        img_data = generate_front_view(img_data)
        
        return lidar_data, img_data
    
    def forward(self, img, lidar, velocity: Optional[torch.Tensor]):
        '''
        Use transfuser to perform data fusion. Call forward method of tranfuser backbone
        
        :param img: list of image inputs
        :param lidar: list of point cloud bev inputs
        :param velocity: scalar of ego velocity. (Optional)
        '''
        use_velocity = self.backbone.use_velocity
        if use_velocity:
            assert isinstance(velocity, torch.Tensor), "velocity must be given as a Tensor!"
        img_feat, lidar_feat, fused_feat = self.backbone(img, lidar, velocity)
        return fused_feat