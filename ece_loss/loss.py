# coding=utf-8
# Copyright (c) Andreas Panteli, Jonas Teuwen
"""Loss module for the Exclusive Cross Entropy."""
import math
from typing import Any, Callable, Optional, Union, Tuple, Type

import torch
from torch import Tensor
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss


def get_sigmoid_annealing_function(epoch_checkpoint: int, temperature: float) -> Callable:
    """
    Implements sigmoid annealing given a checkpoint (for the turning point - middle) and temperature (speed of change)
    """

    def _sigmoid(data):
        """Ordinary sigmoid function"""
        return 1 / (1 + math.exp(-data))

    def _annealing(epoch: int, threshold: float) -> float:
        """Annealing centred around epoch_checkpoint with a decline rate depending on temperature"""
        return _sigmoid((epoch - epoch_checkpoint) / temperature) * threshold

    return _annealing


def get_focal_weighing_function(focal_loss_alpha: float, focal_loss_gamma: float) -> Callable:
    """
    Implements a focal reweighing for loss based on the prediction confidence.
    Assumes loss is pass is defined at it's output as -log(prediction)
    """

    def _focal_weighing(loss: Tensor) -> Tensor:
        loss_p = torch.exp(-loss)
        return focal_loss_alpha * ((1.0 - loss_p) ** focal_loss_gamma) * loss

    return _focal_weighing


class ExclusiveLoss:  # pylint: disable=R0902
    """
    Implements an abstract class of the exclusivity cross entropy as described in the Sparse-shot Learning with
    Exclusive Cross-Entropy for Extremely Many Localisations paper[1].

    Default values are defined as in the original work.

    The different versions, namely BECELoss, BECEWithLogitsLoss, ExclusiveCrossEntropyLoss, implement a similar
    loss as their original pytorch non-exlusive versions for easy/familiar use cases.

    Reference:
        [1] Panteli, A., Teuwen, J., Horlings, H. and Gavves, E.; Sparse-shot Learning with Exclusive Cross-Entropy for
        Extremely Many Localisations; Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),
        2021, pp. 2813-2823
    """

    def __init__(  # pylint: disable=R0913
        self,
        baseloss: Union[Type[BCELoss], Type[BCEWithLogitsLoss], Type[CrossEntropyLoss]],
        target_dtype: torch.dtype,
        confidence_estimator: Callable,
        exclusivity_threshold: float = 0.5,
        background_sampling_threshold: float = 0.5,
        exclusivity_threshold_annealing: Callable = get_sigmoid_annealing_function(50, 10),
        background_sampling_annealing: Callable = get_sigmoid_annealing_function(150, 10),
        focal_loss_parameters: Tuple[float, float] = (0.2, 0.1),
        epoch: int = 0,
    ) -> None:
        # super(ExclusiveLoss, self).__init__()
        self.baseloss = baseloss
        self.target_dtype = target_dtype
        self.confidence_estimator = confidence_estimator
        self.exclusivity_threshold = exclusivity_threshold
        self.background_sampling_threshold = background_sampling_threshold
        self.exclusivity_threshold_annealing = exclusivity_threshold_annealing
        self.background_sampling_annealing = background_sampling_annealing
        self.focal_loss_alpha, self.focal_loss_gamma = focal_loss_parameters
        self.epoch = epoch
        self.unannotated_mapping = self.all_labels_as_unannotated

        self.focal_loss = get_focal_weighing_function(self.focal_loss_alpha, self.focal_loss_gamma)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch at the current state of training for use in the annealing functions"""
        self.epoch = epoch

    @staticmethod
    def get_binary_confidence(predictions: Tensor) -> Tensor:
        """
        For binary predictions, i.e.: object vs. background, the object confidence output is equivalent
        to the prediction confidence. Hence, no change is made.

        Example:
        In a detection task, there will be n bounding boxes with likely properties x,y,w,h,class and confidence, where
        confidence indicates the perceived probability there exists an object/box in this area.
        """
        return predictions

    @staticmethod
    def get_multiclass_confidence(predictions: Tensor) -> Tensor:
        """
        For multiple output class predictions, the confidence is defined as the difference of the top two classes. As
        originally proposed by [1].
        """
        top_2 = torch.topk(predictions, k=2, dim=1)[0]
        predictions = torch.abs(top_2[:, 0, ...] - top_2[:, 1, ...])
        return predictions

    @staticmethod
    def zeroth_label_as_unannotated(targets: Tensor) -> Tensor:
        r"""
        Input of targets must have shape as follows
        :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
        :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of K-dimensional loss.

        :param targets: matrix containing the class labels as integers
        :return: a mask indicating where the unannotated samples are
        """
        return targets == 0

    @staticmethod
    def all_labels_as_unannotated(targets: Tensor) -> Tensor:
        r"""
        Input of targets must have shape as follows
        :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
        :param targets: matrix containing the class labels as integers
        :return: a mask indicating where the unannotated samples are
        """
        return torch.ones(len(targets))

    def set_unannotated_mapping(self, mapping: Callable) -> None:
        """
        Sets the mapping function, given targets, for the unlabelled class
        :param mapping: function to be used when identifying which samples belong to the unannotated cases
        """
        self.unannotated_mapping = mapping

    @staticmethod
    def sample_from_background(predictions: Tensor) -> Tensor:
        """
        Implements sampling across the prediction samples.
        Returns vector values in the range [0, 1] drawn from a uniform distribution.
        """
        return predictions.float().new(predictions.shape).uniform_()

    def forward_(
        self, input: Tensor, target: Tensor, epoch: int, unannotated_mapping: Callable  # pylint: disable=W0622
    ) -> Tensor:

        """
        Uses the exclusivity threshold and annealing functions as described in the Sparse-shot Learning with Exclusive
        Cross-Entropy for Extremely Many Localisations paper[1] to produce the exclusive cross entropy loss.

        :param unannotated_mapping: mapping function to retrieve location of unannotated areas/samples. E.g.: for a
            binary mask this can be a lambda x: x == 0.
        :param input:  representing the unlabelled samples, a list of bounding box confidences of shape BxN, or a list
            of pixels from re-shaped images of shape Bx(HxW), or classification predictions of shape B or multiclass
            input of shape BxC..., where C is the number of classes.
        :param target: label masks of shape BxHxW, or a list of bounding box target confidence of shape BxN, or
            classification targets of shape B.
        :param epoch: the epoch integer value of the training phase
        :return: exclusive cross-entropy
        """
        cross_entropy = self.baseloss_forward(input, target)

        # Background (uniform) sampling. Assuming all input is from the background class
        unannotated_mask = unannotated_mapping(target).bool().to(target.device)
        background_sampling = (
            self.sample_from_background(unannotated_mask)
            <= self.background_sampling_annealing(epoch, self.background_sampling_threshold)
        ) * unannotated_mask

        # Exclusivity threshold
        confidence = self.confidence_estimator(input)
        exclusivity_mask = confidence <= self.exclusivity_threshold_annealing(epoch, self.exclusivity_threshold)
        background_sampling *= exclusivity_mask

        # Focal re-weighing and sampling
        unannotated_loss = self.focal_loss(cross_entropy) * background_sampling
        annotated_loss = cross_entropy * (~unannotated_mask * 1.0)
        ece_loss = unannotated_loss + annotated_loss
        return ece_loss

    def forward(
        self,
        input: Tensor,  # pylint: disable=W0622
        target: Optional[Tensor] = None,
        epoch: Optional[int] = None,
        unannotated_mapping: Optional[Callable] = None,
    ) -> Tensor:
        """
        Overwrites the original forward function of the cross entropy loss (in all its variants: CrossEntropyLoss,
        BCELoss, BCEWithLogitsLoss) in order to be used in the familiar way.

        Assumes default zero tensors for the target and assigns the uses the most updated epoch value (either given
        as input or using the pre-set value)

        :param unannotated_mapping: mapping function to retrieve location of unannotated areas/samples. E.g.: for a
            binary mask this can be a lambda x: x == 0.
        :param epoch: the current epoch number of training used for annealing parameters of the ECE loss
        :param input: representing the unlabelled samples, a list of bounding box confidences of shape BxN, or a list
            of pixels from re-shaped images of shape Bx(HxW), or classification predictions of shape B or multiclass
            input of shape BxC..., where C is the number of classes.
        :param target: Optional parameter for the target ground truth values in the same shape as the expected input
            of the (binary) cross entropy functions. If not given, the assumed background label is zero.
        :return: the exclusive cross entropy loss value
        """
        if target is None:
            target = input.new_zeros(len(input), dtype=self.target_dtype)

        if epoch is None:
            epoch = self.epoch

        if unannotated_mapping is None:
            unannotated_mapping = self.unannotated_mapping

        return self.forward_(input, target, epoch, unannotated_mapping)

    def baseloss_forward(self, input: Tensor, target: Tensor) -> Tensor:  # pylint: disable=W0622
        """Implements the original CE variant forward function"""
        raise NotImplementedError


class BECELoss(ExclusiveLoss, BCELoss):
    """
    Binary Exclusive cross-entropy without activation loss as described in the Sparse-shot Learning with
    Exclusive Cross-Entropy for Extremely Many Localisations paper[1].
    It is a subclass of the abstract class ExclusiveLoss.

    Reference:
        [1] Panteli, A., Teuwen, J., Horlings, H. and Gavves, E., 2021. Sparse-shot Learning with Exclusive
        Cross-Entropy for Extremely Many Localisations. arXiv preprint arXiv:2104.10425.

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as input.

    Example:
        >>> m = nn.Sigmoid()
        >>> loss = nn.BECELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.mean().backward()
    """

    def __init__(  # pylint: disable=R0913
        self,
        weight: Optional[Tensor] = None,
        size_average: Any = None,
        reduce: Any = None,
        reduction: str = "none",
        **kwargs,
    ) -> None:  # pylint: disable=R0801
        BCELoss.__init__(self, weight=weight, size_average=size_average, reduce=reduce, reduction=reduction)
        ExclusiveLoss.__init__(
            self, baseloss=BCELoss, target_dtype=torch.float, confidence_estimator=self.get_binary_confidence, **kwargs
        )

    def baseloss_forward(self, input: Tensor, target: Tensor) -> Tensor:  # pylint: disable=W0622
        return BCELoss.forward(self, input, target)


class BECEWithLogitsLoss(ExclusiveLoss, BCEWithLogitsLoss):
    """
    Binary Exclusive cross-entropy with sigmoid activation loss as described in the Sparse-shot Learning with
    Exclusive Cross-Entropy for Extremely Many Localisations paper[1].
    It is a subclass of the abstract class ExclusiveLoss.

    Reference:
        [1] Panteli, A., Teuwen, J., Horlings, H. and Gavves, E., 2021. Sparse-shot Learning with Exclusive
        Cross-Entropy for Extremely Many Localisations. arXiv preprint arXiv:2104.10425.

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as input.

    Example:
        >>> loss = BECEWithLogitsLoss()
        >>> batch_size = torch.randint(32, (1, 1)).squeeze()
        >>> input = torch.rand(batch_size, requires_grad=True)
        >>> output = loss(input).mean() # Assumes target as a default zero tensor
        >>> output.backward()
    """

    def __init__(  # pylint: disable=R0913
        self,
        weight: Optional[Tensor] = None,
        size_average: Any = None,
        reduce: Any = None,
        reduction: str = "none",
        pos_weight: Optional[Tensor] = None,
        **kwargs,
    ):  # pylint: disable=R0801
        BCEWithLogitsLoss.__init__(
            self, weight=weight, size_average=size_average, reduce=reduce, reduction=reduction, pos_weight=pos_weight
        )
        ExclusiveLoss.__init__(
            self,
            baseloss=BCEWithLogitsLoss,
            target_dtype=torch.float,
            confidence_estimator=self.get_binary_confidence,
            **kwargs,
        )

    def baseloss_forward(self, input: Tensor, target: Tensor) -> Tensor:  # pylint: disable=W0622
        return BCEWithLogitsLoss.forward(self, input, target)


class ExclusiveCrossEntropyLoss(ExclusiveLoss, CrossEntropyLoss):
    r"""
    Exclusive cross-entropy loss as described in the Sparse-shot Learning with Exclusive Cross-Entropy for
    Extremely Many Localisations paper[1]. It is a subclass of the abstract class ExclusiveLoss.

    Reference:
        [1] Panteli, A., Teuwen, J., Horlings, H. and Gavves, E., 2021. Sparse-shot Learning with Exclusive
        Cross-Entropy for Extremely Many Localisations. arXiv preprint arXiv:2104.10425.

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Example:
        >>> loss = ExclusiveCrossEntropyLoss()
        >>> epoch = torch.randint(300, (1, 1)).squeeze()
        >>> batch_size = torch.randint(32, (1, 1)).squeeze()
        >>> num_classes = torch.randint(100, (1, 1)).squeeze()
        >>> input = torch.randn(batch_size, num_classes, requires_grad=True)
        >>> target = torch.empty(batch_size, dtype=torch.long).random_(num_classes)
        >>> output = loss(input, target, epoch)
        >>> output = torch.mean(output)  # If reduction is left to the default of 'none'
        >>> output.backward()
    """

    def __init__(  # pylint: disable=R0913
        self,
        weight: Optional[Tensor] = None,
        size_average: Any = None,
        ignore_index: int = -100,
        reduce: Any = None,
        reduction: str = "none",
        **kwargs,
    ) -> None:  # pylint: disable=R0801
        CrossEntropyLoss.__init__(
            self,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
        )
        ExclusiveLoss.__init__(
            self,
            baseloss=CrossEntropyLoss,
            target_dtype=torch.long,
            confidence_estimator=self.get_multiclass_confidence,
            **kwargs,
        )
        self.set_unannotated_mapping(self.zeroth_label_as_unannotated)

    def baseloss_forward(self, input: Tensor, target: Tensor) -> Tensor:  # pylint: disable=W0622
        return CrossEntropyLoss.forward(self, input, target)
