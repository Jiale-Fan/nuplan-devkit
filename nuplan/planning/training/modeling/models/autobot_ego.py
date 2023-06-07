import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.feature_builders.autobots_feature_builder import AutobotsPredNominalTargetBuilder, AutobotsModeProbsNominalTargetBuilder
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import EgoTrajectoryTargetBuilder

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.autobots_feature_builder import AutobotsMapFeatureBuilder, AutobotsAgentsFeatureBuilder, AutobotsEgoinFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_set_map_feature_builder import VectorSetMapFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.generic_agents_feature_builder import GenericAgentsFeatureBuilder

from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.tensor_target import TensorFeature
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.features.autobots_feature_conversion import NuplanToAutobotsConverter
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.modeling.models.urban_driver_open_loop_model import UrbanDriverOpenLoopModelFeatureParams
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


from nuplan.planning.training.modeling.models.urban_driver_open_loop_model_utils import (
    pad_avails,
    pad_polylines,
)

from typing import List, Optional, cast, Tuple, Union
# from context_encoders import MapEncoderCNN, MapEncoderPts
from nuplan.planning.training.modeling.models.context_encoders import MapEncoderCNN, MapEncoderPts
from nuplan.planning.training.callbacks.utils.visualization_utils import (
    get_raster_from_vector_map_with_agents, get_raster_from_vector_map_with_agents_multiple_trajectories
)
import cv2

def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module






class PositionalEncoding(nn.Module):
    '''
    Standard positional encoding.
    '''
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: must be (T, B, H)
        :return:
        '''
        # example x: [5, 808, 128], self.pe: [20, 1, 128]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# class OutputModel(nn.Module):
#     '''
#     This class operates on the output of AutoBot-Ego's decoder representation. It produces the parameters of a
#     bivariate Gaussian distribution.
#     '''
#     def __init__(self, d_k=64):
#         super(OutputModel, self).__init__()
#         self.d_k = d_k
#         init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
#         self.observation_model = nn.Sequential(
#             init_(nn.Linear(d_k, d_k)), nn.ReLU(),
#             init_(nn.Linear(d_k, d_k)), nn.ReLU(),
#             init_(nn.Linear(d_k, 5))
#         )
#         self.min_stdev = 0.01

#     def forward(self, agent_decoder_state):
#         T = agent_decoder_state.shape[0]
#         BK = agent_decoder_state.shape[1]
#         pred_obs = self.observation_model(agent_decoder_state.reshape(-1, self.d_k)).reshape(T, BK, -1)

#         x_mean = pred_obs[:, :, 0]
#         y_mean = pred_obs[:, :, 1]
#         x_sigma = F.softplus(pred_obs[:, :, 2]) + self.min_stdev
#         y_sigma = F.softplus(pred_obs[:, :, 3]) + self.min_stdev
#         rho = torch.tanh(pred_obs[:, :, 4]) * 0.9  # for stability
#         return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=2)

class OutputModel(nn.Module):
    '''
    This class operates on the output of AutoBot-Ego's decoder representation. It produces the parameters of a
    bivariate Gaussian distribution.
    '''
    def __init__(self, d_k=64):
        super(OutputModel, self).__init__()
        self.d_k = d_k
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.observation_model = nn.Sequential(
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, 5)) # added angle distribution
        )
        self.min_stdev = 0.01

    def forward(self, agent_decoder_state):
        T = agent_decoder_state.shape[0]
        BK = agent_decoder_state.shape[1]
        pred_obs = self.observation_model(agent_decoder_state.reshape(-1, self.d_k)).reshape(T, BK, -1)

        x_mean = pred_obs[:, :, 0]
        y_mean = pred_obs[:, :, 1]
        x_sigma = F.softplus(pred_obs[:, :, 2]) + self.min_stdev
        y_sigma = F.softplus(pred_obs[:, :, 3]) + self.min_stdev
        rho = torch.tanh(pred_obs[:, :, 4]) * 0.9  # for stability
        return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=2)

        # theta_mean = torch.clip(pred_obs[:,:,4], min=-np.pi, max=np.pi)
        # theta_mean = torch.tanh(pred_obs[:,:,4])*np.pi
        # theta_sigma = F.softplus(pred_obs[:, :, 5]) + self.min_stdev

        # return torch.stack([x_mean, y_mean, x_sigma, y_sigma, theta_mean, theta_sigma], dim=2)


class AutoBotEgo(TorchModuleWrapper):
    '''
    AutoBot-Ego Class.
    '''
    def __init__(self, 
        vector_map_feature_radius: int,
        vector_map_connection_scales: Optional[List[int]],
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
        feature_params: UrbanDriverOpenLoopModelFeatureParams,
        d_k=128, _M=5, c=5, T=30, L_enc=1, dropout=0.0, k_attr=2, map_attr=3,
        num_heads=16, L_dec=1, tx_hidden_size=384, use_map_img=False, use_map_lanes=False, 
        draw_visualizations = False
        ):

        self.draw_visualizations = draw_visualizations

        self.converter = NuplanToAutobotsConverter(_M=_M)
        self._feature_params = feature_params

        self.img_num = 0
        
        super().__init__(
            feature_builders=[
                VectorSetMapFeatureBuilder(
                    map_features=feature_params.map_features,
                    max_elements=feature_params.max_elements,
                    max_points=feature_params.max_points,
                    radius=feature_params.vector_set_map_feature_radius,
                    interpolation_method=feature_params.interpolation_method,
                ),
                GenericAgentsFeatureBuilder(feature_params.agent_features, feature_params.past_trajectory_sampling),
            ],
            target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling=future_trajectory_sampling),
             AutobotsPredNominalTargetBuilder(), AutobotsModeProbsNominalTargetBuilder()],
            future_trajectory_sampling=future_trajectory_sampling,
        )

        # super().__init__(
        #     feature_builders=[
        #         VectorMapFeatureBuilder(
        #             radius=vector_map_feature_radius,
        #             connection_scales=vector_map_connection_scales,
        #         ),
        #         AgentsFeatureBuilder(trajectory_sampling=past_trajectory_sampling),
        #     ],
        #     target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling=future_trajectory_sampling),
        #     AutobotsPredNominalTargetBuilder(), AutobotsModeProbsNominalTargetBuilder()],
        #     # target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling=future_trajectory_sampling)],
        #     future_trajectory_sampling=future_trajectory_sampling,
        # )

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.map_attr = map_attr
        self.k_attr = k_attr
        self.d_k = d_k
        self._M = _M  # num agents without the ego-agent
        self.c = c
        self.T = T
        self.L_enc = L_enc
        self.dropout = dropout
        self.num_heads = num_heads
        self.L_dec= L_dec
        self.tx_hidden_size = tx_hidden_size
        self.use_map_img = use_map_img
        self.use_map_lanes = use_map_lanes

        

        # INPUT ENCODERS
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(k_attr, d_k)))

        # ============================== AutoBot-Ego ENCODER ==============================
        self.social_attn_layers = []
        self.temporal_attn_layers = []
        for _ in range(self.L_enc):
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=d_k, nhead=self.num_heads, dropout=self.dropout,
                                                          dim_feedforward=self.tx_hidden_size)
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=d_k, nhead=self.num_heads, dropout=self.dropout,
                                                          dim_feedforward=self.tx_hidden_size)
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)

        # ============================== MAP ENCODER ==========================
        if self.use_map_img:
            self.map_encoder = MapEncoderCNN(d_k=d_k, dropout=self.dropout)
            self.emb_state_map = nn.Sequential(
                    init_(nn.Linear(2 * d_k, d_k)), nn.ReLU(),
                    init_(nn.Linear(d_k, d_k))
                )
        elif self.use_map_lanes:
            self.map_encoder = MapEncoderPts(d_k=d_k, map_attr=map_attr, dropout=self.dropout)
            self.map_attn_layers = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=0.3)

        # ============================== AutoBot-Ego DECODER ==============================
        self.Q = nn.Parameter(torch.Tensor(self.T, 1, self.c, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.Q)

        self.tx_decoder = []
        for _ in range(self.L_dec):
            self.tx_decoder.append(nn.TransformerDecoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                              dropout=self.dropout,
                                                              dim_feedforward=self.tx_hidden_size))
        self.tx_decoder = nn.ModuleList(self.tx_decoder)

        # ============================== Positional encoder ==============================
        self.pos_encoder = PositionalEncoding(d_k, dropout=0.0)

        # ?
        # self.pos_encoder_route = LearntPositionalEncoding(d_k, dropout=0.0)
        # self.pos_encoder_social = PositionalEncoding(d_k, dropout=0.0)

        # ============================== OUTPUT MODEL ==============================
        self.output_model = OutputModel(d_k=self.d_k)

        # ============================== Mode Prob prediction (P(z|X_1:t)) ==============================
        self.P = nn.Parameter(torch.Tensor(c, 1, d_k), requires_grad=True)  # Appendix C.2.
        nn.init.xavier_uniform_(self.P)

        if self.use_map_img:
            self.modemap_net = nn.Sequential(
                init_(nn.Linear(2*self.d_k, self.d_k)), nn.ReLU(),
                init_(nn.Linear(self.d_k, self.d_k))
            )
        elif self.use_map_lanes:
            self.mode_map_attn = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads)

        self.prob_decoder = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)
        self.prob_predictor = init_(nn.Linear(self.d_k, 1))

        self.train()

    def generate_decoder_mask(self, seq_len, device):
        ''' For masking out the subsequent info. '''
        subsequent_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)).bool()
        return subsequent_mask

    def process_observations(self, ego, agents):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [ego, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks_orig = ego[:, :, -1]
        env_masks = (1.0 - env_masks_orig).type(torch.BoolTensor).to(env_masks_orig.device)
        env_masks = env_masks.unsqueeze(1).repeat(1, self.c, 1).view(ego.shape[0] * self.c, -1)

        # Agents stuff
        # agents=agents.cuda()
        temp_masks = torch.cat((torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).type(torch.BoolTensor).to(agents.device)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks, env_masks

    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        num_agents = agent_masks.size(2)
        temp_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        temp_masks[:, -1][temp_masks.sum(-1) == T_obs] = False  # Ensure that agent's that don't exist don't make NaN.
        agents_temp_emb = layer(self.pos_encoder(agents_emb.reshape(T_obs, B * (num_agents), -1)),
                                src_key_padding_mask=temp_masks)
        return agents_temp_emb.view(T_obs, B, num_agents, -1)

    def social_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(self._M + 1, B * T_obs, -1)
        # [TODO]
        # agents_soc_emb = layer(self.pos_encoder(agents_emb), src_key_padding_mask=agent_masks.view(-1, self._M+1))
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.view(-1, self._M+1))
        agents_soc_emb = agents_soc_emb.view(self._M+1, B, T_obs, -1).permute(2, 1, 0, 3)
        return agents_soc_emb


    def extract_agent_features(
        self, ego_agent_features: GenericAgents, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract ego and agent features into format expected by network and build accompanying availability matrix.
        :param ego_agent_features: agent features to be extracted (ego + other agents)
        :param batch_size: number of samples in batch to extract
        :return:
            agent_features: <torch.FloatTensor: batch_size, num_elements (polylines) (1+max_agents*num_agent_types),
                num_points_per_element, feature_dimension>. Stacked ego, agent, and map features.
            agent_avails: <torch.BoolTensor: batch_size, num_elements (polylines) (1+max_agents*num_agent_types),
                num_points_per_element>. Bool specifying whether feature is available or zero padded.
        """
        agent_features = []  # List[<torch.FloatTensor: max_agents+1, total_max_points, feature_dimension>: batch_size]
        agent_avails = []  # List[<torch.BoolTensor: max_agents+1, total_max_points>: batch_size]

        # features have different size across batch so we use per sample feature extraction
        for sample_idx in range(batch_size):
            # Ego features
            # maintain fixed feature size through trimming/padding
            sample_ego_feature = ego_agent_features.ego[sample_idx][
                ..., : min(self._feature_params.ego_dimension, self._feature_params.feature_dimension)
            ].unsqueeze(0)
            if (
                min(self._feature_params.ego_dimension, GenericAgents.ego_state_dim())
                < self._feature_params.feature_dimension
            ):
                sample_ego_feature = pad_polylines(sample_ego_feature, self._feature_params.feature_dimension, dim=2)

            sample_ego_avails = torch.ones(
                sample_ego_feature.shape[0],
                sample_ego_feature.shape[1],
                dtype=torch.bool,
                device=sample_ego_feature.device,
            )

            # reverse points so frames are in reverse chronological order, i.e. (t_0, t_-1, ..., t_-N)
            sample_ego_feature = torch.flip(sample_ego_feature, dims=[1])

            # maintain fixed number of points per polyline
            sample_ego_feature = sample_ego_feature[:, : self._feature_params.total_max_points, ...]
            sample_ego_avails = sample_ego_avails[:, : self._feature_params.total_max_points, ...]
            if sample_ego_feature.shape[1] < self._feature_params.total_max_points:
                sample_ego_feature = pad_polylines(sample_ego_feature, self._feature_params.total_max_points, dim=1)
                sample_ego_avails = pad_avails(sample_ego_avails, self._feature_params.total_max_points, dim=1)

            sample_features = [sample_ego_feature]
            sample_avails = [sample_ego_avails]

            # Agent features
            agent_types_num = len(self._feature_params.agent_features)
            for i, feature_name in enumerate(self._feature_params.agent_features):
                # if there exist at least one valid agent in the sample
                if ego_agent_features.has_agents(feature_name, sample_idx):
                    # num_frames x num_agents x num_features -> num_agents x num_frames x num_features
                    sample_agent_features = torch.permute(
                        ego_agent_features.agents[feature_name][sample_idx], (1, 0, 2)
                    )

                    # concat agent type one-hot encoding to agent features
                    one_hot_encodings = torch.zeros((sample_agent_features.shape[0], sample_agent_features.shape[1], agent_types_num), device=sample_agent_features.device)
                    one_hot_encodings[:, :, i] = 1
                    sample_agent_features = torch.cat((sample_agent_features, one_hot_encodings), dim=-1)

                    # maintain fixed feature size through trimming/padding
                    sample_agent_features = sample_agent_features[
                        ..., : min(self._feature_params.agent_dimension, self._feature_params.feature_dimension)
                    ]
                    if (
                        min(self._feature_params.agent_dimension, GenericAgents.agents_states_dim())
                        < self._feature_params.feature_dimension
                    ):
                        sample_agent_features = pad_polylines(
                            sample_agent_features, self._feature_params.feature_dimension, dim=2
                        )

                    sample_agent_avails = torch.ones(
                        sample_agent_features.shape[0],
                        sample_agent_features.shape[1],
                        dtype=torch.bool,
                        device=sample_agent_features.device,
                    )

                    # reverse points so frames are in reverse chronological order, i.e. (t_0, t_-1, ..., t_-N)
                    sample_agent_features = torch.flip(sample_agent_features, dims=[1])

                    # maintain fixed number of points per polyline
                    sample_agent_features = sample_agent_features[:, : self._feature_params.total_max_points, ...]
                    sample_agent_avails = sample_agent_avails[:, : self._feature_params.total_max_points, ...]
                    if sample_agent_features.shape[1] < self._feature_params.total_max_points:
                        sample_agent_features = pad_polylines(
                            sample_agent_features, self._feature_params.total_max_points, dim=1
                        )
                        sample_agent_avails = pad_avails(
                            sample_agent_avails, self._feature_params.total_max_points, dim=1
                        )

                    # maintained fixed number of agent polylines of each type per sample
                    sample_agent_features = sample_agent_features[: self._feature_params.max_agents, ...]
                    sample_agent_avails = sample_agent_avails[: self._feature_params.max_agents, ...]
                    if sample_agent_features.shape[0] < (self._feature_params.max_agents):
                        sample_agent_features = pad_polylines(
                            sample_agent_features, self._feature_params.max_agents, dim=0
                        )
                        sample_agent_avails = pad_avails(sample_agent_avails, self._feature_params.max_agents, dim=0)

                else:
                    sample_agent_features = torch.zeros(
                        self._feature_params.max_agents,
                        self._feature_params.total_max_points,
                        self._feature_params.feature_dimension,
                        dtype=torch.float32,
                        device=sample_ego_feature.device,
                    )
                    sample_agent_avails = torch.zeros(
                        self._feature_params.max_agents,
                        self._feature_params.total_max_points,
                        dtype=torch.bool,
                        device=sample_agent_features.device,
                    )

                # add features, avails to sample
                sample_features.append(sample_agent_features)
                sample_avails.append(sample_agent_avails)

            sample_features = torch.cat(sample_features, dim=0)
            sample_avails = torch.cat(sample_avails, dim=0)

            agent_features.append(sample_features)
            agent_avails.append(sample_avails)
        agent_features = torch.stack(agent_features)
        agent_avails = torch.stack(agent_avails)

        return agent_features, agent_avails

    def extract_map_features(
        self, vector_set_map_data: VectorSetMap, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract map features into format expected by network and build accompanying availability matrix.
        :param vector_set_map_data: VectorSetMap features to be extracted
        :param batch_size: number of samples in batch to extract
        :return:
            map_features: <torch.FloatTensor: batch_size, num_elements (polylines) (max_lanes),
                num_points_per_element, feature_dimension>. Stacked map features.
            map_avails: <torch.BoolTensor: batch_size, num_elements (polylines) (max_lanes),
                num_points_per_element>. Bool specifying whether feature is available or zero padded.
        """
        map_features = []  # List[<torch.FloatTensor: max_map_features, total_max_points, feature_dim>: batch_size]
        map_avails = []  # List[<torch.BoolTensor: max_map_features, total_max_points>: batch_size]

        # features have different size across batch so we use per sample feature extraction
        for sample_idx in range(batch_size):

            sample_map_features = []
            sample_map_avails = []

            for feature_name in self._feature_params.map_features:
                coords = vector_set_map_data.coords[feature_name][sample_idx]
                tl_data = (
                    vector_set_map_data.traffic_light_data[feature_name][sample_idx]
                    if feature_name in vector_set_map_data.traffic_light_data
                    else None
                )
                avails = vector_set_map_data.availabilities[feature_name][sample_idx]

                # add traffic light data if exists for feature
                if tl_data is not None:
                    coords = torch.cat((coords, tl_data), dim=2)

                # maintain fixed number of points per map element (polyline)
                coords = coords[:, : self._feature_params.total_max_points, ...]
                avails = avails[:, : self._feature_params.total_max_points]

                if coords.shape[1] < self._feature_params.total_max_points:
                    coords = pad_polylines(coords, self._feature_params.total_max_points, dim=1)
                    avails = pad_avails(avails, self._feature_params.total_max_points, dim=1)

                # maintain fixed number of features per point
                coords = coords[..., : self._feature_params.feature_dimension]
                if coords.shape[2] < self._feature_params.feature_dimension:
                    coords = pad_polylines(coords, self._feature_params.feature_dimension, dim=2)

                sample_map_features.append(coords)
                sample_map_avails.append(avails)

            map_features.append(torch.cat(sample_map_features))
            map_avails.append(torch.cat(sample_map_avails))

        map_features = torch.stack(map_features)
        map_avails = torch.stack(map_avails)

        return map_features, map_avails

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "tensor_map": Tensor,
                            "agents": Agents,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                            "mode_probs": TensorTarget(data=mode_probs), 
                            "pred": TensorTarget(data=out_dists)
                        }
        """
        vector_set_map_data = cast(VectorSetMap, features["vector_set_map"])
        ego_agent_features = cast(GenericAgents, features["generic_agents"])
        batch_size = ego_agent_features.batch_size

        # Extract features across batch
        agent_features, agent_avails = self.extract_agent_features(ego_agent_features, batch_size)
        map_in, map_avails = self.extract_map_features(vector_set_map_data, batch_size)


        #     agent_features: <torch.FloatTensor: batch_size, num_elements (polylines) (1+max_agents*num_agent_types),
        #         num_points_per_element, feature_dimension>. Stacked ego, agent, and map features.
        #     agent_avails: <torch.BoolTensor: batch_size, num_elements (polylines) (1+max_agents*num_agent_types),
        #         num_points_per_element>. Bool specifying whether feature is available or zero padded.
        #     map_features: <torch.FloatTensor: batch_size, num_elements (polylines) (max_lanes),
        #         num_points_per_element, feature_dimension>. Stacked map features.
        #     map_avails: <torch.BoolTensor: batch_size, num_elements (polylines) (max_lanes),
        #         num_points_per_element>. Bool specifying whether feature is available or zero padded.



        # roads = cast(TensorFeature, features["tensor_map"]).data
        # agents = cast(Agents, features["agents"])
        # agents_in = self.converter.AgentsToAutobotsAgentsTensor(agents)
        # ego_in= self.converter.AgentsToAutobotsEgoinTensor(agents)

        roads = torch.cat((map_in, map_avails.unsqueeze(-1)), dim=3) # [8, 160, 20, 9]
        agents_in_and_ego = torch.cat((agent_features, agent_avails.unsqueeze(-1)), dim=3).transpose(1, 2)  # agent features' feature dimension from 3 to 8 are padded with 0s.
        ego_in = agents_in_and_ego[:, :, 0, :] # [8, 20, 9]
        agents_in = agents_in_and_ego[:, :, 1:, :] # [8, 20, 30, 9]
        
        
        '''
        :param ego_in: [B, T_obs, k_attr+1] with last values being the existence mask. 
        :param agents_in: [B, T_obs, M-1, k_attr+1] with last values being the existence mask.
        :param roads: [B, S, P, map_attr+1] representing the road network if self.use_map_lanes or
                      [B, 3, 128, 128] image representing the road network if self.use_map_img or
                      [B, 1, 1] if self.use_map_lanes and self.use_map_img are False.
        :return:
            pred_obs: shape [c, T, B, 5] c trajectories for the ego agents with every point being the params of
                                        Bivariate Gaussian distribution.
            mode_probs: shape [B, c] mode probability predictions P(z|X_{1:T_obs})
        '''
        # ego_in [64,4,3]
        # agents_in [64,4,7,3]
        # roads [64,100,40,4] [Batch, Segment, Points, attributes ()]
        # B should be batch
        # T_obs should be observation Time (input time) 
        # k_attr should be the number of the attributes at one timestamp, namely x, y, mask
        B = ego_in.size(0)

        # Encode all input observations (k_attr --> d_k)
        ego_tensor, _agents_tensor, opps_masks, env_masks = self.process_observations(ego_in, agents_in)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2)
        agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)

        # Process through AutoBot's encoder
        for i in range(self.L_enc):
            agents_emb = self.temporal_attn_fn(agents_emb, opps_masks, layer=self.temporal_attn_layers[i])
            agents_emb = self.social_attn_fn(agents_emb, opps_masks, layer=self.social_attn_layers[i])
        ego_soctemp_emb = agents_emb[:, :, 0]  # take ego-agent encodings only.

        # Process map information
        if self.use_map_img:
            orig_map_features = self.map_encoder(roads)
            map_features = orig_map_features.view(B * self.c, -1).unsqueeze(0).repeat(self.T, 1, 1)
        elif self.use_map_lanes:
            # [TODO] only add the ego agent's route with learned positional encoding.
            # 
            # roads = self.pos_encoder_route(roads)
            orig_map_features, orig_road_segs_masks = self.map_encoder(roads, ego_soctemp_emb)
            map_features = orig_map_features.unsqueeze(2).repeat(1, 1, self.c, 1).view(-1, B*self.c, self.d_k)
            road_segs_masks = orig_road_segs_masks.unsqueeze(1).repeat(1, self.c, 1).view(B*self.c, -1)

        # Repeat the tensors for the number of modes for efficient forward pass.
        context = ego_soctemp_emb.unsqueeze(2).repeat(1, 1, self.c, 1)
        context = context.view(-1, B*self.c, self.d_k)

        # AutoBot-Ego Decoding
        out_seq = self.Q.repeat(1, B, 1, 1).view(self.T, B*self.c, -1)
        time_masks = self.generate_decoder_mask(seq_len=self.T, device=ego_in.device)
        for d in range(self.L_dec):
            if self.use_map_img and d == 1:
                ego_dec_emb_map = torch.cat((out_seq, map_features), dim=-1)
                out_seq = self.emb_state_map(ego_dec_emb_map) + out_seq
            elif self.use_map_lanes and d == 1:
                ego_dec_emb_map = self.map_attn_layers(query=out_seq, key=map_features, value=map_features,
                                                       key_padding_mask=road_segs_masks)[0]
                out_seq = out_seq + ego_dec_emb_map
            out_seq = self.tx_decoder[d](out_seq, context, tgt_mask=time_masks, memory_key_padding_mask=env_masks)
        out_dists = self.output_model(out_seq).reshape(self.T, B, self.c, -1).permute(2, 0, 1, 3)

        # Mode prediction
        mode_params_emb = self.P.repeat(1, B, 1)
        mode_params_emb = self.prob_decoder(query=mode_params_emb, key=ego_soctemp_emb, value=ego_soctemp_emb)[0]
        if self.use_map_img:
            mode_params_emb = self.modemap_net(torch.cat((mode_params_emb, orig_map_features.transpose(0, 1)), dim=-1))
        elif self.use_map_lanes:
            mode_params_emb = self.mode_map_attn(query=mode_params_emb, key=orig_map_features, value=orig_map_features,
                                                 key_padding_mask=orig_road_segs_masks)[0] + mode_params_emb
        mode_probs = F.softmax(self.prob_predictor(mode_params_emb).squeeze(-1), dim=0).transpose(0, 1)

        traj=self.converter.output_tensor_to_trajectory(out_dists, mode_probs)

        # return  [c, T, B, 5], [B, c]
        # return out_dists, mode_probs

        multimodal_trajectories = self.converter.output_distribution_to_multimodal_trajectories(out_dists)
        
        if self.draw_visualizations:
            multimodal_traj_draw = Trajectory(data=multimodal_trajectories.squeeze(0))
            image_ndarray = get_raster_from_vector_map_with_agents_multiple_trajectories(vector_set_map_data.to_device('cpu'),
                                                                    ego_agent_features.to_device('cpu'), 
                                                                    target_trajectory=None,
                                                    predicted_trajectory=multimodal_traj_draw.to_device('cpu'), 
                                                    pixel_size=0.1)
            cv2.imwrite(f"/data1/nuplan/jiale/exp/autobots_experiment/images/multimodal_vis_{self.img_num:04d}.png", image_ndarray)
            self.img_num += 1

        return {"trajectory": traj, "mode_probs": TensorFeature(data=mode_probs), "pred": TensorFeature(data=out_dists),
                "multimodal_trajectories": TensorFeature(data=multimodal_trajectories)}
        # return {"trajectory": traj}


# def pack_multimodal_trajectories(out_dists: Tensor) -> List[Trajectory]:
#     """
#         out_dists: Tensor [c, T, B, 5]
#     """
#     reshaped_out_dists = out_dists.permute(2, 0, 1, 3)  # [B, c, T, 5]
#     B = reshaped_out_dists.size(0)
#     multimodal_trajectories = []
#     for b in range(B):
#         traj_data = reshaped_out_dists[b, :, :, :3]
#         traj_data[:,:,-1] = 0 # all angles being zero [TODO]
#         traj = Trajectory(data=traj_data)
#         multimodal_trajectories.append(traj)

#     return multimodal_trajectories