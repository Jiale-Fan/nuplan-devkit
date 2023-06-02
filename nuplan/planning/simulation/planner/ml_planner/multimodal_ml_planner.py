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
from nuplan.planning.training.preprocessing.features.tensor_target import TensorFeature

from nuplan.planning.training.modeling.objectives.trajectory_metric_eval_utils import *
from torch import Tensor
from unittest.mock import Mock, patch
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

from nuplan.planning.script.builders.metric_builder import build_metrics_engines_planner
from omegaconf import DictConfig, OmegaConf
from nuplan.common.actor_state.ego_state import EgoState

from nuplan.planning.simulation.callback.metric_callback import run_metric_engine_planner

class MultimodalMLPlanner(AbstractPlanner):
    """
    Implements abstract planner interface.
    Used for simulating any ML planner trained through the nuPlan training framework.
    """

    def __init__(self, model: TorchModuleWrapper) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """
        self._future_horizon = model.future_trajectory_sampling.time_horizon
        self._step_interval = model.future_trajectory_sampling.step_time
        self._num_output_dim = model.future_trajectory_sampling.num_poses

        self._model_loader = ModelLoader(model)

        self._initialization: Optional[PlannerInitialization] = None

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

        yaml_str = """

                    low_level: # Low level metrics
                        - ego_lane_change_statistics
                        - ego_jerk_statistics
                        - ego_lat_acceleration_statistics
                        - ego_lon_acceleration_statistics
                        - ego_lon_jerk_statistics
                        - ego_yaw_acceleration_statistics
                        - ego_yaw_rate_statistics
                        
                    high_level:  # High level metrics that depend on low level metrics, they can also rely on the previously called high level metrics
                        - drivable_area_compliance_statistics
                        - speed_limit_compliance_statistics
                        - ego_is_comfortable_statistics
                        - driving_direction_compliance_statistics
                """

        # Parse the YAML string into a DictConfig object
        self.metric_config = OmegaConf.create(yaml_str)

    def _infer_model(self, features: FeaturesType, current_ego_state: EgoState) -> npt.NDArray[np.float32]:
        """
        Makes a single inference on a Pytorch/Torchscript model.

        :param features: dictionary of feature types
        :return: predicted trajectory poses as a numpy array
        """
        # Propagate model
        predictions = self._model_loader.infer(features)

        # Extract trajectory prediction
        trajectory_predicted = cast(TensorFeature, predictions['multimodal_trajectories'])
        trajectories_tensor = trajectory_predicted.data

        trajectories_object = Trajectory(trajectories_tensor)

        selected_trajectory = self._select_best_trajectory(trajectories_object, current_ego_state)

        trajectory = selected_trajectory.data.cpu().detach().numpy()[0]  # retrive first (and only) batch as a numpy array

        return cast(npt.NDArray[np.float32], trajectory)

    def _select_best_trajectory(self, trajectories: Trajectory, current_ego_state: EgoState) -> Trajectory:
        """
        Selects the best trajectory from a set of trajectories.
        
        param: trajectories: a Trajectory object containing a batch of trajectories
        param: current_ego_state: the current ego state
        return: the best trajectory
        """
        # [TODO] mock a scenario, pack map_api into it. 
        mock_scenario = Mock(spec=NuPlanScenario)
        mock_scenario.map_api = self._initialization.map_api
        mock_scenario.initial_ego_state = current_ego_state
        
        metrics_list = []

        traj_list = trajectories.unpack()
        for trajectory in traj_list:
            simulation_history = get_simulation_history(trajectory, mock_scenario)
            metric_engine = build_metrics_engines_planner(self.metric_config, mock_scenario)
            metrics_list.append(run_metric_engine_planner(metric_engine, mock_scenario, "autobot", simulation_history))

        # [TODO] select the best trajectory based on the metrics
        return trajectories[0]
    
    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._model_loader.initialize()
        self._initialization = initialization

        

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        # Extract history
        history = current_input.history

        # Construct input features
        start_time = time.perf_counter()
        features = self._model_loader.build_features(current_input, self._initialization)
        self._feature_building_runtimes.append(time.perf_counter() - start_time)

        # Infer model
        start_time = time.perf_counter()
        predictions = self._infer_model(features, ego_state=history.ego_states[-1])
        self._inference_runtimes.append(time.perf_counter() - start_time)

        # Convert relative poses to absolute states and wrap in a trajectory object.
        states = transform_predictions_to_states(
            predictions, history.ego_states, self._future_horizon, self._step_interval
        )
        trajectory = InterpolatedTrajectory(states)

        return trajectory

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []
        return report
