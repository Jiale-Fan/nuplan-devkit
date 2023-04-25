from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import Tensor

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.torch_geometry import coordinates_to_local_frame
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario


from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import EgoTrajectoryTargetBuilder
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.tensor_target import TensorFeature

from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder

from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.features.autobots_feature_conversion import NuplanToAutobotsConverter
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

import numpy as np
from numpy.typing import NDArray
import itertools

# def coords_to_map_attr(coords) -> NDArray:
#     """Map coordinates in VectorMap format to AutoBots features

#     Args:
#         coords NDArray with shape [num_segments, 2, 2]: one element in coords List of VectorMap

#     Returns:
#         point_feature_tab NDArray with shape [num_segments, 4]: 4 attributes are x, y, angles, existence mask
#     """
#     vec = np.squeeze(coords[:,1,:])-np.squeeze(coords[:,0,:])
#     angles=np.arctan2(vec[:,1], vec[:,0]).reshape((-1, 1))

#     # get point feature tablular of shape [p_total, 3]
#     point_feature_tab=np.concatenate((np.squeeze(coords[:,0,:]), angles), axis=1)
#     point_feature_tab=np.pad(point_feature_tab, ((0, 0), (0, 1)), "constant", constant_values=(1))

#     # [TODO] omit the first point as [0, 0, 0], greatly simplify the process
#     point_feature_tab[0,:]=np.zeros((1,4)) 
#     return point_feature_tab


# This class is unused
class AutobotsMapFeatureBuilder(VectorMapFeatureBuilder):

    def __init__(self, radius: float, converter: NuplanToAutobotsConverter, connection_scales: Optional[List[int]] = None) -> None:
        """
        Initialize vector map builder with configuration parameters.
        :param radius:  The query radius scope relative to the current ego-pose.
        :param connection_scales: Connection scales to generate. Use the 1-hop connections if it's left empty.
        :return: Vector map data including lane segment coordinates and connections within the given range.
        """
        super().__init__(radius, connection_scales)
        self.converter = converter

    @torch.jit.unused
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return TensorFeature # type: ignore

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "tensor_map"

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> Tensor:
        vec_map=super(AutobotsMapFeatureBuilder, self).get_features_from_scenario(scenario)
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        tf=TensorFeature(data=self.converter.VectorMapToAutobotsMapTensor(vec_map))
        return tf

    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> Tensor:
        vec_map=super(AutobotsMapFeatureBuilder, self).get_features_from_simulation(current_input, initialization)
        return TensorFeature(data=self.converter.VectorMapToAutobotsMapTensor(vec_map))


# This class is unused
class AutobotsAgentsFeatureBuilder(AgentsFeatureBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling, converter: NuplanToAutobotsConverter) -> None:
        """
        Initializes AgentsFeatureBuilder.
        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        """
        super().__init__(trajectory_sampling)
        self.converter = converter


    @torch.jit.unused
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return Tensor # type: ignore

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "tensor_agents"

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> Tensor:
        agent=super(AutobotsAgentsFeatureBuilder, self).get_features_from_scenario(scenario)
        return TensorFeature(data=self.converter.AgentsToAutobotsAgentsTensor(agent))

    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> Tensor:
        agent=super(AutobotsAgentsFeatureBuilder, self).get_features_from_simulation(current_input, initialization)
        return TensorFeature(data=self.converter.AgentsToAutobotsAgentsTensor(agent))


class AutobotsEgoinFeatureBuilder(AgentsFeatureBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling, converter: NuplanToAutobotsConverter) -> None:
        """
        Initializes AgentsFeatureBuilder.
        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        """
        super().__init__(trajectory_sampling)
        self.converter = converter


    @torch.jit.unused
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return Tensor # type: ignore

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "tensor_egoin"

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> Tensor:
        agent=super(AutobotsEgoinFeatureBuilder, self).get_features_from_scenario(scenario)
        return TensorFeature(data=self.converter.AgentsToAutobotsEgoinTensor(agent))

    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> Tensor:
        agent=super(AutobotsEgoinFeatureBuilder, self).get_features_from_simulation(current_input, initialization)
        return TensorFeature(data=self.converter.AgentsToAutobotsEgoinTensor(agent))

    

# This class is unused
# class AutobotsTargetBuilder(EgoTrajectoryTargetBuilder):
#     """Trajectory builders constructed the desired ego's trajectory from a scenario."""
    
#     def __init__(self, future_trajectory_sampling: TrajectorySampling, converter: NuplanToAutobotsConverter) -> None:
#         """
#         Initializes EgoTrajectoryTargetBuilder.
#         :param future_trajectory_sampling: Parameters of the sampled trajectory of the ego vehicle
#         """
#         super().__init__(future_trajectory_sampling)
#         self.converter = converter
    
#     @classmethod
#     def get_feature_unique_name(cls) -> str:
#         """Inherited, see superclass."""
#         return "tensor_trajectory"

#     @classmethod
#     def get_feature_type(cls) -> Type[AbstractModelFeature]:
#         """Inherited, see superclass."""
#         return Tensor  # type: ignore

#     def get_targets(self, scenario: AbstractScenario) -> Tensor:
#         targets = super(AutobotsTargetBuilder, self).get_targets(scenario)

#         # since in AutoBots, the last colomn of values are existence mask, not heading direction angles, 
#         # we overwrite them all with 1

#         targets=self.converter.TrajectoryToAutobotsTarget(targets)

#         return Tensor(targets)

#     def TrajectoryToAutobotsTarget(self, target: Trajectory) -> Tensor:
#         """_summary_

#         Args:
#             target (Trajectory): attribute data: either a [num_batches, num_states, 3] or [num_states, 3] representing the trajectory
#                  where se2_state is [x, y, heading] with units [meters, meters, radians].

#         Returns:
#             Tensor: _description_
#         """

#         target_ts=torch.as_tensor(target.data)
#         if target_ts.dim() == 2:
#             target_ts=torch.unsqueeze(target_ts, 0) # if two dimension, unsqueeze to create one more "batch" dimension
#         target_ts[:,:,2]=0
#         return target_ts

class AutobotsPredNominalTargetBuilder(AbstractTargetBuilder):
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "pred"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return TensorFeature  # type: ignore

    def get_targets(self, scenario: AbstractScenario) -> Tensor:

        nominal_target = TensorFeature(data=np.zeros((2,2)))
        return nominal_target


class AutobotsModeProbsNominalTargetBuilder(AbstractTargetBuilder):
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "mode_probs"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return TensorFeature  # type: ignore

    def get_targets(self, scenario: AbstractScenario) -> Tensor:

        nominal_target = TensorFeature(data=np.zeros((2,2)))
        return nominal_target
       