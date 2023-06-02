from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from hydra.utils import instantiate

from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import StepSimulationTimeController
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation

from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup

from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from typing import cast

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from torch import Tensor

from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.common.actor_state.state_representation import TimeDuration
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
import numpy as np
import numpy.typing as npt
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


# def get_reward(trajectories: Trajectory, scenarios: ScenarioListType, metrics: List[MetricBase]):
#     """
#     Get reward for a batch of trajectories.
#     :param trajectories: Batch of trajectories.
#     :param scenarios: Batch of scenarios.
#     :param metrics: List of metrics.
#     :return: Reward.
#     """

#     metric_score_list = [ get_metric_scores(trajectories, scenarios, metric) for metric in metrics ]
#     # ??????????????????
#     reward_list = [ metric_score.mean() for metric_score in metric_score_list ]
#     return Tensor(reward_list)

# def planned_trajectory_to_simulation_history(trajectory: Trajectory, scenario: AbstractScenario) -> SimulationHistory:
#     """
#     Fake one planned trajectory to a simulation history.
#     Note that only extract_ego_states and map_api are valid in the simulation history.

#     Args:
#         trajectory (Trajectory): planned trajectory
#         scenario (AbstractScenario): scenario

#     Returns:
#         SimulationHistory: simulation history for metrics evaluation
#     """

#     history = SimulationHistory(scenario.map_api, scenario.get_mission_goal(), )
#     return history

def get_metric_scores(trajectories: Trajectory, scenarios: ScenarioListType, metric: MetricBase, close_loop = False) -> Tensor:

    score_list = []
    for trajectory, scenario in zip(trajectories, scenarios):
        history = get_simulation_history(trajectory, scenario, time_horizon = 8.0, close_loop=close_loop)
        score_list.append( get_single_metric_score(history, scenario, metric))
    return Tensor(score_list)
        
def get_metric_scores(trajectories: Trajectory, scenario: AbstractScenario, metric: MetricBase, close_loop = False) -> Tensor:

    score_list = []
    for trajectory in trajectories:
        history = get_simulation_history(trajectory, scenario, time_horizon = 8.0, close_loop=close_loop)
        score_list.append( get_single_metric_score(history, scenario, metric))
    return Tensor(score_list)

def get_single_metric_score(history: SimulationHistory, scenario: AbstractScenario, metric: MetricBase) -> float:
    """
    Get metric term for a trajectory.
    :param trajectory: Trajectory.
    :param scenario: Scenario.
    :param metric: Metric.
    :return: Metric score. !! Make sure the metric's compute score function is implemented!
    """

    # stat_list = metric.compute(history, scenario) # return List[MetricStatistics]
    
    # tricky part: high level statistics require low level statistics to be computed first, but
    # if not, it won't report any exception but will return invalid results
    # [TODO] currently this only applies to DrivableAreaComplianceStatistics
    metric._lane_change_metric.compute(history, scenario)
    metrics_statistics = metric.compute(history, scenario) # return List[MetricStatistics]

    # extract the metric score from the statistics
    # [TODO]
    score = metrics_statistics[0].metric_score
    return score


def get_simulation_history(trajectory: Trajectory, scenario: AbstractScenario, time_horizon: float = 8.0, close_loop = False) -> SimulationHistory:
    
    simulation = get_simulation(scenario, close_loop=close_loop)

    simulation.initialize()
    begin_time = simulation._time_controller.get_iteration().time_point
    planner_input = simulation.get_planner_input()

    trajectory_tensor = trajectory.data
    trajectory_np = trajectory_tensor.cpu().detach().numpy()[0]  # retrive first (and only) batch as a numpy array
    trajectory_np = cast(npt.NDArray[np.float32], trajectory_np)

    states = transform_predictions_to_states(
        trajectory_np, planner_input.history.ego_states, future_horizon = 8.0, step_interval = 0.5)
    trajectory_ip = InterpolatedTrajectory(states)

    # todo: add variable future time horizon
    while simulation._time_controller.get_iteration().time_point.diff(begin_time) < TimeDuration.from_s(time_horizon-0.6):
        # since the simulation steps after the time check, we make a 0.1s safe margin, otherwise the simulation may step over the time horizon
        simulation.propagate(trajectory_ip)

    history = simulation.history
    return history


def get_simulation(scenario: AbstractScenario, close_loop = False) -> Simulation:
        
    # ego_controller: AbstractEgoController = instantiate(cfg.ego_controller, scenario=scenario)
    ego_controller = PerfectTrackingController(scenario=scenario)

    # Simulation Manager
    # simulation_time_controller: AbstractSimulationTimeController = instantiate(
    #     cfg.simulation_time_controller, scenario=scenario
    # )
    simulation_time_controller = StepSimulationTimeController(scenario=scenario)

    # Perception
    if not close_loop:
        observations=TracksObservation(scenario=scenario)
    else:
        observations = instantiate(config_name = "idm_agents_observation")

    # Construct simulation and manager
    simulation_setup = SimulationSetup(
        time_controller=simulation_time_controller,
        observations=observations,
        ego_controller=ego_controller,
        scenario=scenario,
    )

    simulation = Simulation(
        simulation_setup=simulation_setup,
        callback=None,
        simulation_history_buffer_duration=2.0,
    )

    return simulation