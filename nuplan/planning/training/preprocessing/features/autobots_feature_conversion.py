from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import Tensor


from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.features.agents import Agents

from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

import numpy as np

def coords_to_map_attr(coords) -> Tensor:
    """Map coordinates in VectorMap format to AutoBots features

    Args:
        coords torch.Tensor with shape [num_segments, 2, 2]: one element in coords List of VectorMap

    Returns:
        point_feature_tab torch.Tensor with shape [num_segments, 4]: 4 attributes are x, y, angles, existence mask
    """

    vec = torch.squeeze(coords[:, 1, :]) - torch.squeeze(coords[:, 0, :])
    
    angles = torch.atan2(vec[:, 1], vec[:, 0]).reshape((-1, 1))

    # get point feature tabular of shape [p_total, 3]
    point_feature_tab = torch.cat((torch.squeeze(coords[:, 0, :]), angles), dim=1)
    # pad the last dimension with existence mask all being 1
    point_feature_tab = torch.nn.functional.pad(point_feature_tab, (0, 1), "constant", value=1)

    # [NOTE] omit the first point as [0, 0, 0, 0], which greatly simplify the process and should not affect
    # too much the training
    point_feature_tab[0, :] = torch.zeros((1, 4))
    return point_feature_tab


@torch.jit.unused 
def VectorMapToAutobotsMapTensor(vec_map: VectorMap) -> Tensor:
    """_summary_

    Args:
        vec_map: lane_groupings: List[List[<np.ndarray: num_lane_segments_in_lane>]].
            Each lane grouping or polyline is represented by an array of indices of lane segments
            in coords belonging to the given lane. Each batch contains a List of lane groupings.

    Returns:
        Tensor: shape [B, S, P, map_attr+1] example [64,100,40,4]
    """
    
    # B = len(vec_map.coords) # get the number of batches

    # TODO: S and P dimension must be bigger than that of the original data. 
    # Adapting dimension? 
    S = 200 # 100
    P = 600 # 40
    
    # to debug: check the maximum number of segment contained in one lane
    lengths = [[len(x) for x in sublist] for sublist in vec_map.lane_groupings]
    length_maxes = [torch.max(torch.tensor(x)) for x in lengths]
    max_p_num = torch.max(torch.tensor(length_maxes))

    # pad the lane_groupings
    # padded_list_list = [[torch.nn.functional.pad(x, (0, max(P - len(x), 0)), 'constant', 0) for x in sublist] for sublist in vec_map.lane_groupings]
    list_idx = [torch.nn.utils.rnn.pad_sequence(sublist, batch_first=True) for sublist in vec_map.lane_groupings]
    # list_idx shape = List[Tensor[num_lane(varies), max_p_num(varies)]] 
    padded_list_idx = [torch.nn.functional.pad(x[:, :min(x.shape[1], P)], (0, max(P-x.shape[1], 0)), 'constant', 0) for x in list_idx] # map_autobots shape is [B, S, P, 4]
    # list_idx shape = List[Tensor[num_lane(varies), P]]

    list_of_feature_array = [ coords_to_map_attr(coord_mat) for coord_mat in vec_map.coords]

    lane_features = [ feature[idx.long()] for idx, feature in zip(padded_list_idx, list_of_feature_array)]  # List[Tensor(num_lane, P, 4)]
    lane_features_tensor = torch.nn.utils.rnn.pad_sequence(lane_features, batch_first=True) # Tensor(B, num_lane, num_p, 4)

    lf_shape=lane_features_tensor.shape
    map_autobots =  torch.nn.functional.pad(lane_features_tensor[:, :min(lf_shape[1], S), :, :], (0, 0, 0, 0, 0, max(S-lf_shape[1], 0)), 'constant', 0) # map_autobots shape is [B, S, P, 4]

    return map_autobots


@torch.jit.unused
def AgentsToAutobotsAgentsTensor(agents: Agents) -> Tensor:
    """_summary_

    Args:
        agents
        The structure inludes:
        ego: List[<np.ndarray: num_frames, 3>].
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The last dimension is the ego pose (x, y, heading) at time t.
            Example dimensions: 8 (batch_size) x 5 (1 present + 4 past frames) x 3
        agents: List[<np.ndarray: num_frames, num_agents, 8>].
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The num_agents is padded to fit the largest number of agents across all frames.
            The last dimension is the agent pose (x, y, heading) velocities (vx, vy, yaw rate)
             and size (length, width) at time t.

    Returns:
        Tensor: [B, T_obs, M-1, k_attr+1] example [64,4,7,3]
    """

    # every scenario may have different number of agents
    M_minus_1 = 100  # maximum agent number

    lengths = [x.shape[1] for x in agents.agents]
    length_maxes = [torch.max(torch.tensor(x)) for x in lengths]
    max_length = torch.max(torch.tensor(length_maxes))

    # agents.agents: List[Tensor[T_obs, num_agents(varies), 8]]
    padded_list = [torch.nn.functional.pad(x, (0, 0, 0, max(M_minus_1 - x.shape[1],0)), 'constant') for x in agents.agents]
    # padded_list: List[Tensor[T_obs, M-1, 8]]
    
    agents_ts = torch.stack(padded_list)
    agents_ts = agents_ts[:, :, :, :3]  # take only x, y coordinates and an additional dimension to be existence mask
    agents_ts[:, :, :, 2] = (agents_ts[:, :, :, 2] != 0).float() # modify the last column to be existence mask

    return agents_ts


@torch.jit.unused 
def AgentsToAutobotsEgoinTensor(agents: Agents):
    """_summary_

    Args:
        agents
        The structure inludes:
        ego: List[<np.ndarray: num_frames, 3>].
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The last dimension is the ego pose (x, y, heading) at time t.
            Example dimensions: 8 (batch_size) x 5 (1 present + 4 past frames) x 3
        agents: List[<np.ndarray: num_frames, num_agents, 8>].
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The num_agents is padded to fit the largest number of agents across all frames.
            The last dimension is the agent pose (x, y, heading) velocities (vx, vy, yaw rate)
             and size (length, width) at time t.

    Returns:
        Tensor: [B, T_obs, k_attr+1] example [64,4,3]
    """

    ego_in=torch.stack(agents.ego)
    ego_in[:,:,2]=1
    return ego_in

@torch.jit.unused
def TrajectoryToAutobotsEgoin(traj: Trajectory) -> Tensor:
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