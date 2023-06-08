import logging
import pathlib
from typing import Dict, List

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
import os
import hydra


logger = logging.getLogger(__name__)

CONFIG_PATH = os.getenv('NUPLAN_HYDRA_CONFIG_PATH', 'config/simulation')

if os.environ.get('NUPLAN_HYDRA_CONFIG_PATH') is not None:
    CONFIG_PATH = os.path.join('../../../../', CONFIG_PATH)

if os.path.basename(CONFIG_PATH) != 'simulation':
    CONFIG_PATH = os.path.join(CONFIG_PATH, 'simulation')
CONFIG_NAME = 'default_simulation'


def build_high_level_metric(cfg: DictConfig, base_metrics: Dict[str, AbstractMetricBuilder]) -> AbstractMetricBuilder:
    """
    Build a high level metric.
    :param cfg: High level metric config.
    :param base_metrics: A dict of base metrics.
    :return A high level metric.
    """
    # Make it editable
    OmegaConf.set_struct(cfg, False)
    required_metrics: Dict[str, str] = cfg.pop('required_metrics', {})
    OmegaConf.set_struct(cfg, True)

    metric_params = {}
    for metric_param, metric_name in required_metrics.items():
        metric_params[metric_param] = base_metrics[metric_name]

    return instantiate(cfg, **metric_params)


def build_metrics_engines(cfg: DictConfig, scenarios: List[AbstractScenario]) -> Dict[str, MetricsEngine]:
    """
    Build a metric engine for each different scenario type.
    :param cfg: Config.
    :param scenarios: list of scenarios for which metrics should be build.
    :return Dict of scenario types to metric engines.
    """
    main_save_path = pathlib.Path(cfg.output_dir) / cfg.metric_dir

    # Metrics selected by user
    selected_metrics = cfg.selected_simulation_metrics
    if isinstance(selected_metrics, str):
        selected_metrics = [selected_metrics]

    simulation_metrics = cfg.simulation_metric
    low_level_metrics: DictConfig = simulation_metrics.get('low_level', {})
    high_level_metrics: DictConfig = simulation_metrics.get('high_level', {})

    metric_engines = {}
    for scenario in scenarios:
        # If we already have the engine for the specific scenario type, we can skip it
        if scenario.scenario_type in metric_engines:
            continue
        # Metrics
        metric_engine = MetricsEngine(main_save_path=main_save_path, timestamp=cfg.experiment_time)

        # TODO: Add scope checks
        scenario_type = scenario.scenario_type
        scenario_metrics: DictConfig = simulation_metrics.get(scenario_type, {})
        metrics_in_scope = low_level_metrics.copy()
        metrics_in_scope.update(scenario_metrics)

        high_level_metric_in_scope = high_level_metrics.copy()
        # We either pick the selected metrics if any is specified, or all metrics
        if selected_metrics is not None:
            metrics_in_scope = {
                metric_name: metrics_in_scope[metric_name]
                for metric_name in selected_metrics
                if metric_name in metrics_in_scope
            }
            high_level_metric_in_scope = {
                metric_name: high_level_metrics[metric_name]
                for metric_name in selected_metrics
                if metric_name in high_level_metric_in_scope
            }
        base_metrics = {
            metric_name: instantiate(metric_config) for metric_name, metric_config in metrics_in_scope.items()
        }

        for metric in base_metrics.values():
            metric_engine.add_metric(metric)

        # Add high level metrics
        for metric_name, metric in high_level_metric_in_scope.items():
            high_level_metric = build_high_level_metric(cfg=metric, base_metrics=base_metrics)
            metric_engine.add_metric(high_level_metric)

            # Add the high-level metric to the base metrics, so that other high-level metrics can reuse it
            base_metrics[metric_name] = high_level_metric

        metric_engines[scenario_type] = metric_engine

    return metric_engines


def build_metrics_engines_planner(metrics_config: DictConfig, scenario: AbstractScenario) -> MetricsEngine:
    """
    Build a metric engine for each different scenario type.
    :param cfg: Config.
    :param scenario: current mocked scenario.
    :return Dict of scenario types to metric engines.
    """
    main_save_path = pathlib.Path("./cache/planner_metric_cache")

    low_level_metrics: DictConfig = metrics_config.get('low_level', {})
    high_level_metrics: DictConfig = metrics_config.get('high_level', {})

    # Metrics
    metric_engine = MetricsEngine(main_save_path=main_save_path, timestamp=0)

    # TODO: Add scope checks
    # scenario_type = scenario.scenario_type
    # scenario_metrics: DictConfig = metric_config.get(scenario_type, {})
    # metrics_in_scope = low_level_metrics.copy()
    # metrics_in_scope.update(scenario_metrics)

    metrics_in_scope = low_level_metrics.copy()

    high_level_metric_in_scope = high_level_metrics.copy()
    # We either pick the selected metrics if any is specified, or all metrics

    base_metrics = {
        metric_name: instantiate(metric_config) for metric_name, metric_config in metrics_in_scope.items()
    }

    for metric in base_metrics.values():
        metric_engine.add_metric(metric)

    # Add high level metrics
    for metric_name, metric in high_level_metric_in_scope.items():
        high_level_metric = build_high_level_metric(cfg=metric, base_metrics=base_metrics)
        metric_engine.add_metric(high_level_metric)

        # Add the high-level metric to the base metrics, so that other high-level metrics can reuse it
        base_metrics[metric_name] = high_level_metric

    return metric_engine



PLANNER_METRICS_CONFIG = '''

low_level: # Low level metrics
    ego_lane_change_statistics:
        _target_: nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change.EgoLaneChangeStatistics
        _convert_: 'all'
        name: 'ego_lane_change'
        category: 'Planning'
        max_fail_rate: 0.3


    ego_jerk_statistics:
        _target_: nuplan.planning.metrics.evaluation_metrics.common.ego_jerk.EgoJerkStatistics
        _convert_: 'all'
        name: 'ego_jerk'
        category: 'Dynamics'

        max_abs_mag_jerk: 8.37

    ego_lat_acceleration_statistics:
        _target_: nuplan.planning.metrics.evaluation_metrics.common.ego_lat_acceleration.EgoLatAccelerationStatistics
        _convert_: 'all'
        name: 'ego_lat_acceleration'
        category: 'Dynamics'

        max_abs_lat_accel: 4.89

    ego_lon_acceleration_statistics:
        _target_: nuplan.planning.metrics.evaluation_metrics.common.ego_lon_acceleration.EgoLonAccelerationStatistics
        _convert_: 'all'
        name: 'ego_lon_acceleration'
        category: 'Dynamics'

        min_lon_accel: -4.05
        max_lon_accel: 2.40

    ego_lon_jerk_statistics:
        _target_: nuplan.planning.metrics.evaluation_metrics.common.ego_lon_jerk.EgoLonJerkStatistics
        _convert_: 'all'
        name: 'ego_lon_jerk'
        category: 'Dynamics'

        max_abs_lon_jerk: 4.13

    ego_yaw_acceleration_statistics:
        _target_: nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_acceleration.EgoYawAccelerationStatistics
        _convert_: 'all'
        name: 'ego_yaw_acceleration'
        category: 'Dynamics'

        max_abs_yaw_accel: 1.93

    ego_yaw_rate_statistics:
        _target_: nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_rate.EgoYawRateStatistics
        _convert_: 'all'
        name: 'ego_yaw_rate'
        category: 'Dynamics'

        max_abs_yaw_rate: 0.95


high_level:  # High level metrics that depend on low level metrics, they can also rely on the previously called high level metrics
    drivable_area_compliance_statistics:
        _target_: nuplan.planning.metrics.evaluation_metrics.common.drivable_area_compliance.DrivableAreaComplianceStatistics
        _convert_: 'all'
        name: 'drivable_area_compliance'
        category: 'Planning'
        metric_score_unit: 'bool'

        max_violation_threshold: 0.3 # The violatation tolerance threshold in meters

        required_metrics:
            # Parameter: base metric name and other high level metrics used in this metric
            lane_change_metric: ego_lane_change_statistics

    # speed_limit_compliance_statistics:
    #     _target_: nuplan.planning.metrics.evaluation_metrics.common.speed_limit_compliance.SpeedLimitComplianceStatistics
    #     _convert_: 'all'
    #     name: 'speed_limit_compliance'
    #     category: 'Violations'
    #     metric_score_unit: 'float'
    #     max_violation_threshold: 1.0
    #     max_overspeed_value_threshold: 2.23

    #     required_metrics:
    #         # Parameter: base metric name and other high level metrics used in this metric
    #         lane_change_metric: ego_lane_change_statistics

    ego_is_comfortable_statistics:
        _target_: nuplan.planning.metrics.evaluation_metrics.common.ego_is_comfortable.EgoIsComfortableStatistics
        _convert_: 'all'
        name: 'ego_is_comfortable'
        category: 'Violations'
        metric_score_unit: 'bool'

        required_metrics:
            # Parameter: base metric name
            ego_jerk_metric: ego_jerk_statistics
            ego_lat_acceleration_metric: ego_lat_acceleration_statistics
            ego_lon_acceleration_metric: ego_lon_acceleration_statistics
            ego_lon_jerk_metric: ego_lon_jerk_statistics
            ego_yaw_acceleration_metric: ego_yaw_acceleration_statistics
            ego_yaw_rate_metric: ego_yaw_rate_statistics


    driving_direction_compliance_statistics:
        _target_: nuplan.planning.metrics.evaluation_metrics.common.driving_direction_compliance.DrivingDirectionComplianceStatistics
        _convert_: 'all'
        name: 'driving_direction_compliance'
        category: 'Planning'
        metric_score_unit: 'bool'

        driving_direction_compliance_threshold: 2 # [m] Driving in opposite direction up to this threshold isn't considered violation
        driving_direction_violation_threshold: 6 # [m] Driving in opposite direction above this threshold isn't tolerated
        time_horizon: 1 # [s] time horizon in which movement of the vehicle along baseline direction is computed.

        required_metrics:
            # Parameter: base metric name and other high level metrics used in this metric
            lane_change_metric: ego_lane_change_statistics

    speed_limit_compliance_statistics:
        _target_: nuplan.planning.metrics.evaluation_metrics.common.speed_limit_compliance.SpeedLimitComplianceStatistics
        _convert_: 'all'
        name: 'speed_limit_compliance'
        category: 'Violations'
        metric_score_unit: 'float'
        max_violation_threshold: 1.0
        max_overspeed_value_threshold: 2.23

        required_metrics:
            # Parameter: base metric name and other high level metrics used in this metric
            lane_change_metric: ego_lane_change_statistics

    ego_mean_speed_statistics:
        _target_: nuplan.planning.metrics.evaluation_metrics.common.ego_mean_speed.EgoMeanSpeedStatistics
        _convert_: 'all'
        name: 'ego_mean_speed'
        category: 'Dynamics'


'''
