from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torchvision
from numpy import ndarray
from torch import Tensor

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature, FeatureDataType

@dataclass
class ImageLidar(AbstractModelFeature):
    """
    Dataclass of mixed image and lidar feature. Held in the form of (HxWxC) or (CxHxW) etc.
    
    :param TODO: haven't decided the eventual structure of data
    """
    
    data: FeatureDataType
    
    @property
    def num_batches(self) -> Optional[int]:
        return None if len(self.data.shape) < 4 else self.data.shape[0]
    
    def to_feature_tensor(self) -> AbstractModelFeature:
        to_tensor_torchvision = torchvision.transforms.ToTensor()
        return ImageLidar(data=to_tensor_torchvision(np.asarray(self.data)))
    
    def to_device(self, device: torch.device):
        validate_type(self.data, torch.Tensor)
        return ImageLidar(data=self.data.to(device))
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> ImageLidar:
        return ImageLidar(data=data['data'])
    
    def unpack(self):
        pass
        
    def from_feature_tensor(self):
        pass
    
    @property
    def width(self) -> int:
        """
        :return: the width of a raster
        """
        return self.data.shape[-2] if self._is_channels_last() else self.data.shape[-1]  # type: ignore

    @property
    def height(self) -> int:
        """
        :return: the height of a raster
        """
        return self.data.shape[-3] if self._is_channels_last() else self.data.shape[-2]  # type: ignore

    def num_channels(self) -> int:
        """
        Number of raster channels.
        """
        return self.data.shape[-1] if self._is_channels_last() else self.data.shape[-3]  # type: ignore
    
    def _is_channels_last(self) -> bool:
        """
        Check location of channel dimension
        :return True if position [-1] is the number of channels
        """
        # For tensor, channel is put right before the spatial dimention.
        if isinstance(self.data, Tensor):
            return False

        # The default numpy array data format is channel last.
        elif isinstance(self.data, ndarray):
            return True
        else:
            raise RuntimeError(
                f'The data needs to be either numpy array or torch Tensor, but got type(data): {type(self.data)}'
            )

    def _get_data_channel(self, index: Union[int, range]) -> FeatureDataType:
        """
        Extract channel data
        :param index: of layer
        :return: data corresponding to layer
        """
        if self._is_channels_last():
            return self.data[..., index]
        else:
            return self.data[..., index, :, :]
        
        
# The mixture feature builder takes in the image and lidar pc for feature fusion, and return the model with fused feature type EnvToken as env token.
# So, the EnvTokenBuilder should contains both parts of processing image and lidar data.