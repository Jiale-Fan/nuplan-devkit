from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import Tensor

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.torch_geometry import coordinates_to_local_frame
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.feature_builders.scriptable_feature_builder import ScriptableFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    LaneOnRouteStatusData,
    LaneSegmentConnections,
    LaneSegmentCoords,
    LaneSegmentGroupings,
    LaneSegmentTrafficLightData,
    get_neighbor_vector_map,
    get_on_route_status,
    get_traffic_light_encoding,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import EgoTrajectoryTargetBuilder
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

import numpy as np
from numpy.typing import NDArray
import itertools

def coords_to_map_attr(coords) -> NDArray:
    """Map coordinates in VectorMap format to AutoBots features

    Args:
        coords NDArray with shape [num_segments, 2, 2]: one element in coords List of VectorMap

    Returns:
        point_feature_tab NDArray with shape [num_segments, 4]: 4 attributes are x, y, angles, existence mask
    """
    coords=coords.cpu()
    vec = np.squeeze(coords[:,1,:])-np.squeeze(coords[:,0,:])
    
    angles=np.arctan2(vec[:,1], vec[:,0]).reshape((-1, 1))

    # get point feature tablular of shape [p_total, 3]
    point_feature_tab=np.concatenate((np.squeeze(coords[:,0,:]), angles), axis=1)
    point_feature_tab=np.pad(point_feature_tab, ((0, 0), (0, 1)), "constant", constant_values=(1))

    # [TODO] omit the first point as [0, 0, 0], greatly simplify the process
    point_feature_tab[0,:]=np.zeros((1,4)) 
    return point_feature_tab

@torch.jit.unused 
def VectorMapToAutobotsMapTensor(vec_map: VectorMap):
    """_summary_

    Args:
        scenario (AbstractScenario): see base class

    Returns:
        Tensor: shape [B, S, P, map_attr+1] example [64,100,40,4]
    """
    
    # B=len(vec_map.coords) # get the number of batches

    # TODO: S and P dimension must be bigger than that of the original data. 
    # Adapting dimension? 
    S=200 # 100
    P=600 # 40
    
    # to debug: check the maximum number of segment contained in one lane
    lengths = [[len(x) for x in sublist] for sublist in vec_map.lane_groupings]
    length_maxes=[ np.max(np.array(x)) for x in lengths]
    max_length = np.max(length_maxes)

    padded_list_list = [[np.pad(x.cpu(), (0, max(P - len(x), 0)), 'constant') for x in sublist] for sublist in vec_map.lane_groupings]
    # [TODO]if P < len(x) ??
    # if you experience exception here, it may be the presence of P < len(x). Check max_length and P values.
    list_of_idx_array = [ np.array(l, np.float64) for l in padded_list_list] # l's shape = [num_lane, P]
    list_of_feature_array = [ coords_to_map_attr(coord_mat) for coord_mat in vec_map.coords]

    lane_features = [ feature[idx.astype(np.int32)] for idx, feature in zip(list_of_idx_array, list_of_feature_array)]

    # ((pad_top, pad_bottom), (pad_left, pad_right))
    padded_list = [ np.pad(arr, ((0, S-arr.shape[0]), (0, 0), (0, 0)), 'constant')  for arr in lane_features] # get list of array of shape [S, P]

    map_autobots=np.array(padded_list, np.float64) # map_autobots shape is [B, S, P, 4]

    return Tensor(map_autobots)


@torch.jit.unused 
def AgentsToAutobotsAgentsTensor(agents: Agents):
    """_summary_

    Args:
        scenario (AbstractScenario): see base class

    Returns:
        Tensor: [B, T_obs, M-1, k_attr+1] example [64,4,7,3]
    """

    # every scenario may have different number of agents
    M_minus_1 = 80 # maximum agent number

    lengths = [x.shape[1] for x in agents.agents]
    length_maxes=[ np.max(np.array(x)) for x in lengths]
    max_length = np.max(length_maxes)

    padded_list=[ np.pad(arr.cpu(), ((0, 0), (0, max(M_minus_1-arr.shape[1], 0)), (0, 0)), 'constant')  for arr in agents.agents]

    extended_list= [np.expand_dims(x, 0) for x in padded_list]
    
    agents_ts= np.concatenate(extended_list, 0)
    
    agents_ts=agents_ts[:,:,:,0:3] # take only x, y coordinates and an addtional dimension to be existence mask

    agents_ts[:,:,:,2] = (agents_ts[:,:,:,2]!=0)


    return Tensor(agents_ts)

@torch.jit.unused 
def AgentsToAutobotsEgoinTensor(agents: Agents):
    """_summary_

    Args:
        scenario (AbstractScenario): see base class

    Returns:
        Tensor: [B, T_obs, k_attr+1] example [64,4,3]
    """

    ego_in=torch.stack(agents.ego)
    ego_in[:,:,2]=1
    return ego_in

@torch.jit.unused
def TrajectoryToAutobotsEgoin(traj: Trajectory):
    target_ts=traj.data
    target_ts[:,:,2]=1
    return target_ts

@torch.jit.unused
def output_tensor_to_trajectory(pred_obs: Tensor, mode_probs: Tensor) -> Trajectory:
    """_summary_

    Args:
        pred_obs: shape [c, T, B, 5] c trajectories for the ego agents with every point being the params of
                                        Bivariate Gaussian distribution.
        mode_probs: shape [B, c] mode probability predictions P(z|X_{1:T_obs})
    """

    most_likely_idx=torch.argmax(mode_probs, 1)
    # for each batch, pick the trajectory with largest probability
    trajs=torch.stack([pred_obs[most_likely_idx[i],:,i,:] for i in range(pred_obs.shape[2])])

    trajs_3=trajs[:,:,:3]
    trajs_3[:,:,2] = 1

    return Trajectory(data=trajs_3)