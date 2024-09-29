from click import Choice
import torch
import time
from typing import List, Optional, Type, cast

import numpy as np
import numpy.typing as npt

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.planner.ml_planner.model_loader import ModelLoader
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

class TrajectoryDecoder(TorchModuleWrapper):
    '''
    The decoder that takes as input the env_tokens and decode the predicted trajectory using trajectory vocabulary.
    In detail, the decoder is responsible for computing the attention score of traj_query with env_tokens, the metric score and pick the final trajectory. To do so, the decoder need to first convert the input features using MLP. See Hydra-MDP for details.
    '''
    
    def __init__(self, voc_size, voc_path, embed_size):
        '''
        :param voc_size: the size of trajectory vocabulary, which is predefined using kMeans
        :param voc_path: the path to the vocabulary in shape of (size, num_states=40, 3). numpy.array
        :param embed_size: the hidden size of traj_voc embeddings, which is produced by MLP + Transformer
        :decoder_config: config for constructing the decoder
        '''
        pass

    def initialize(self, ):
        pass

    def traj_voc_embedding(self, voc: Choice[torch.Tensor, np.array]):
        '''
        Construct the embedding of traj voc. Performed by an MLP and a transformer encoder with self-attention
        '''
        pass

    def ego_status_embedding(self, ego_status):
        '''
        Construct the embedding of ego_status. Notice that this is used in every inference.
        '''
        pass

    def forward(self):
        pass

    