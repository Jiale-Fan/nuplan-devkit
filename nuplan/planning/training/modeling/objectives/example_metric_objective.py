from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import extract_scenario_type_weight
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.tensor_target import TensorFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

from nuplan.planning.training.modeling.objectives.autobots_train_helpers import nll_loss_multimodes, nll_loss_multimodes_joint
from torch import Tensor
from nuplan.planning.training.modeling.objectives.metric_objective_utils import get_metric_scores
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase

class ExampleMetricObjective(AbstractObjective):
    """
    Objective utilizing metrics from the nuplan library
    """

    def __init__(self, scenario_type_loss_weighting: Dict[str, float], entropy_weight, kl_weight, use_FDEADE_aux_loss, metric: MetricBase):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'example_metric_objective'
        self.entropy_weight=entropy_weight
        self.kl_weight=kl_weight
        self.use_FDEADE_aux_loss=use_FDEADE_aux_loss
        self._scenario_type_loss_weighting = scenario_type_loss_weighting
        self._metric = metric

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        
        """
        pred_obs = cast(TensorFeature, predictions["pred"]).data
        mode_probs = cast(TensorFeature, predictions["mode_probs"]).data
        trajectories = cast(Trajectory, targets["trajectory"])
        trajectories_tensor = trajectories.data

        metric_scores = get_metric_scores(trajectories, scenarios, self._metric)


        # loss_weights = extract_scenario_type_weight(
        #     scenarios, self._scenario_type_loss_weighting, device=pred_obs.device
        # ) # [B]
        

        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs, trajectories_tensor[:, :, :2], mode_probs,
                                                                                   entropy_weight=self.entropy_weight,
                                                                                   kl_weight=self.kl_weight,
                                                                                   use_FDEADE_aux_loss=self.use_FDEADE_aux_loss)

        total_loss=nll_loss + adefde_loss + kl_loss # scalar
        # how to implement the gradient clip?

        # nll_loss: 

        return total_loss
