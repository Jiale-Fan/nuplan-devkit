from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import extract_scenario_type_weight
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.tensor_target import TensorFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

from nuplan.planning.training.modeling.objectives.autobots_train_helpers import nll_loss_multimodes, nll_loss_multimodes_joint, scenario_specific_loss
from torch import Tensor

from torch.nn import functional as F

class AutobotsObjective(AbstractObjective):
    """
    Autobots ego objective
    """

    def __init__(self, scenario_type_loss_weighting: Dict[str, float], entropy_weight, kl_weight, use_FDEADE_aux_loss, cross_entropy_weight):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'autobots_ego_objective'
        self.entropy_weight=entropy_weight
        self.kl_weight=kl_weight
        self.use_FDEADE_aux_loss=use_FDEADE_aux_loss
        self.cross_entropy_weight = cross_entropy_weight

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
        scenario_type_probs = cast(TensorFeature, predictions["scenario_type"]).data


        targets_xy = cast(Trajectory, targets["trajectory"]).data
        scenario_types = cast(TensorFeature, targets["scenario_type"]).data # (batch_size)
        
        self.cross_entropy_weight *= 0.999

        cls_loss = F.cross_entropy(scenario_type_probs, scenario_types)

        # fde_ade_loss, _, weighted_cls_loss  = scenario_specific_loss(pred_obs, targets_xy[:, :, :2], mode_probs, scenario_types, self.cross_entropy_weight)
        # total_loss = fde_ade_loss + weighted_cls_loss


        nll_loss, kl_loss, post_entropy, adefde_loss = \
            nll_loss_multimodes(select_traj(pred_obs, scenario_types, 6), targets_xy[:, :, :2], 
                mode_probs[torch.arange(mode_probs.shape[0]), scenario_types],
                entropy_weight=self.entropy_weight,
                kl_weight=self.kl_weight,
                use_FDEADE_aux_loss=self.use_FDEADE_aux_loss)

        total_loss=nll_loss + adefde_loss + kl_loss + self.cross_entropy_weight*cls_loss
        
        return total_loss

# def select_mode_prebs(mode_pred, scenario_type, mode_per_scenario) -> Tensor:
#     """_summary_

#     Args:
#         mode_pred (_type_): [B, K*m]
#         scenario_type (_type_): [B]

#     return:
#         mode_probs (_type_): [B, K]
#     """

#     B = mode_pred.shape[0]

#     mode_probs = torch.zeros((B, mode_per_scenario), device=mode_pred.device)

#     for b in range(B):
#         mode_probs[b,:] = mode_pred[b, scenario_type[b]*mode_per_scenario:(scenario_type[b]+1)*mode_per_scenario]

#     return mode_probs


def select_traj(pred, scenario_type, mode_per_scenario) -> Tensor:
    """_summary_

    Args:
        pred (_type_): [K*m,T,B,5]
        scenario_type (_type_): [B]

    return:
        trajs (_type_): [K,T,B,5]
    """

    K = pred.shape[0]
    T = pred.shape[1]
    B = pred.shape[2]

    trajs = torch.zeros((mode_per_scenario,T,B,5), device=pred.device)


    for b in range(B):
        trajs[:,:,b,:] = pred[scenario_type[b]*mode_per_scenario:(scenario_type[b]+1)*mode_per_scenario,:,b,:]

    return trajs