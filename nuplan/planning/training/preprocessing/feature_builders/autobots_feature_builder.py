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
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
import numpy as np
from numpy.typing import NDArray

def coords_to_map_attr(coords) -> NDArray:
    """_summary_

    Args:
        coords (_type_): _description_

    Returns:
        _type_: _description_
    """
    vec = np.squeeze(coords[:,1,:])-np.squeeze(coords[:,0,:])
    angles=np.arctan2(vec[:,1], vec[:,0])

    # get point feature tablular of shape [p_total, 3]
    point_feature_tab=np.concatenate((np.squeeze(coords[:,0,:]), angles), axis=1)
    return point_feature_tab



class AutobotsMapFeatureBuilder(VectorMapFeatureBuilder):
    @torch.jit.unused
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return Tensor # type: ignore

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> Tensor:
        """_summary_

        Args:
            scenario (AbstractScenario): _description_

        Returns:
            Tensor: shape [B, S, P, map_attr+1] example [64,100,40,4]
        """
       
        vec_map=super(AutobotsMapFeatureBuilder, self).get_features_from_scenario(scenario)
        B=len(vec_map.coords) # get the number of batches

        # TODO: If we need to determine those dimensions in other ways
        S=100
        P=40
        
        padded_list_list = [[np.pad(x, (0, P - len(x)), 'constant') for x in sublist] for sublist in vec_map.lane_groupings]
        # [TODO]if P < len(x) ??
        list_of_idx_array = [ np.array(l, np.float64) for l in padded_list_list] # l's shape = [num_lane, P]
        list_of_feature_array = [ coords_to_map_attr(coord_mat) for coord_mat in vec_map.coords]

        lane_features = [ feature(idx) for idx, feature in zip(list_of_idx_array, list_of_feature_array)]

        # ((pad_top, pad_bottom), (pad_left, pad_right))
        padded_list = [ np.pad(arr, ((0, S-arr.shape[0]), (0, 0), (0, 0)), 'constant')  for arr in lane_features] # get list of array of shape [S, P]

        map_autobots=np.array(padded_list, np.float64) # map_autobots shape is [B, S, P, 4]


        # TODO:  How to address the mask zero

        return map_autobots




    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> VectorMap:
        """Inherited, see superclass."""
        pass