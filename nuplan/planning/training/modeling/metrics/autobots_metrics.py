
from typing import Dict, List, cast
import torch

from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric
from nuplan.planning.training.modeling.types import TargetsType
from nuplan.planning.training.preprocessing.features.tensor_target import TensorFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

from nuplan.planning.training.modeling.objectives.autobots_train_helpers import nll_loss_multimodes, scenario_specific_loss


class AutobotsNllLoss(AbstractTrainingMetric):
    """
    
    """

    def __init__(self, entropy_weight, kl_weight, use_FDEADE_aux_loss):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'nll_loss'
        self.entropy_weight=entropy_weight
        self.kl_weight=kl_weight
        self.use_FDEADE_aux_loss=use_FDEADE_aux_loss

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory", "pred", "mode_probs"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        
        """

        pred_obs = cast(TensorFeature, predictions["pred"]).data
        mode_probs = cast(TensorFeature, predictions["mode_probs"]).data
        targets_xy = cast(Trajectory, targets["trajectory"]).data
        

        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs, targets_xy[:, :, :2], mode_probs,
                                                                                   entropy_weight=self.entropy_weight,
                                                                                   kl_weight=self.kl_weight,
                                                                                   use_FDEADE_aux_loss=self.use_FDEADE_aux_loss)

        return nll_loss


class AutobotsKlLoss(AbstractTrainingMetric):
    """
    
    """

    def __init__(self, entropy_weight, kl_weight, use_FDEADE_aux_loss):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'kl_loss'
        self.entropy_weight=entropy_weight
        self.kl_weight=kl_weight
        self.use_FDEADE_aux_loss=use_FDEADE_aux_loss

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory", "pred", "mode_probs"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        
        """

        pred_obs = cast(TensorFeature, predictions["pred"]).data
        mode_probs = cast(TensorFeature, predictions["mode_probs"]).data
        targets_xy = cast(Trajectory, targets["trajectory"]).data
        

        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs, targets_xy[:, :, :2], mode_probs,
                                                                                   entropy_weight=self.entropy_weight,
                                                                                   kl_weight=self.kl_weight,
                                                                                   use_FDEADE_aux_loss=self.use_FDEADE_aux_loss)

        return kl_loss

class AutobotsPostEntropy(AbstractTrainingMetric):
    """
    
    """

    def __init__(self, entropy_weight, kl_weight, use_FDEADE_aux_loss):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'post_entropy'
        self.entropy_weight=entropy_weight
        self.kl_weight=kl_weight
        self.use_FDEADE_aux_loss=use_FDEADE_aux_loss

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory", "pred", "mode_probs"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        
        """

        pred_obs = cast(TensorFeature, predictions["pred"]).data
        mode_probs = cast(TensorFeature, predictions["mode_probs"]).data
        targets_xy = cast(Trajectory, targets["trajectory"]).data
        

        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs, targets_xy[:, :, :2], mode_probs,
                                                                                   entropy_weight=self.entropy_weight,
                                                                                   kl_weight=self.kl_weight,
                                                                                   use_FDEADE_aux_loss=self.use_FDEADE_aux_loss)

        return post_entropy

class AutobotsADEFDELoss(AbstractTrainingMetric):
    """
    
    """

    def __init__(self, entropy_weight, kl_weight, use_FDEADE_aux_loss):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'ade_fde_loss'
        self.entropy_weight=entropy_weight
        self.kl_weight=kl_weight
        self.use_FDEADE_aux_loss=use_FDEADE_aux_loss

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory", "pred", "mode_probs"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        
        """

        pred_obs = cast(TensorFeature, predictions["pred"]).data
        mode_probs = cast(TensorFeature, predictions["mode_probs"]).data
        targets_xy = cast(Trajectory, targets["trajectory"]).data
        

        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs, targets_xy[:, :, :2], mode_probs,
                                                                                   entropy_weight=self.entropy_weight,
                                                                                   kl_weight=self.kl_weight,
                                                                                   use_FDEADE_aux_loss=self.use_FDEADE_aux_loss)

        return adefde_loss

class AutobotsScenarioClassificationLoss(AbstractTrainingMetric):
    """
    
    """

    def __init__(self):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'cross_entropy_loss'

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["scenario_type", "mode_probs"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        
        """

        mode_probs = cast(TensorFeature, predictions["scenario_type"]).data
        scenario_types = cast(TensorFeature, targets["scenario_type"]).data
        
        cls_loss_fn = torch.nn.CrossEntropyLoss()
        cls_loss = cls_loss_fn(mode_probs, scenario_types)

        return cls_loss


class AutobotsScenarioADEFDELoss(AbstractTrainingMetric):
    """
    
    """

    def __init__(self):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'scenario_ade_fde_loss'

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["scenario_type", "mode_probs"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        
        """
        pred_obs = cast(TensorFeature, predictions["pred"]).data
        mode_probs = cast(TensorFeature, predictions["mode_probs"]).data
        targets_xy = cast(Trajectory, targets["trajectory"]).data
        scenario_types = cast(TensorFeature, targets["scenario_type"]).data
        
        
        fde_ade_loss, _, _ = scenario_specific_loss(pred_obs, targets_xy[:, :, :2], mode_probs, scenario_types)

        return fde_ade_loss