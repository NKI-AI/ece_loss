import torch
from torch import Tensor
from typing import Callable, Optional
from torch.nn.modules.loss import CrossEntropyLoss


class ExclusiveCrossEntropyLoss(CrossEntropyLoss):
    """
    Implements the exclusivity cross entropy as described in the Sparse-shot Learning with Exclusive
    Cross-Entropy for Extremely Many Localisations paper[1].

    Reference:
        [1] Panteli, A., Teuwen, J., Horlings, H. and Gavves, E., 2021. Sparse-shot Learning with Exclusive
        Cross-Entropy for Extremely Many Localisations. arXiv preprint arXiv:2104.10425.
    """
    def __init__(self,
                 background_threshold: Optional[float] = 0.75,
                 sampling_threshold: Optional[float] = 0.5,
                 multiclass: Optional[bool] = False,
                 annealing_function_background: Optional[Callable] = lambda x: x,
                 annealing_function_sampling: Optional[Callable] = lambda x: x,
                 focal_loss_parameters: Optional[tuple[float, float]] = (0.2, 0.1),
                 *args, **kwargs) -> None:
        super(ExclusiveCrossEntropyLoss, self).__init__(reduction='none', *args, **kwargs)
        self.ignore_index = super().ignore_index
        self.background_threshold = background_threshold
        self.sampling_threshold = sampling_threshold
        self.multiclass = multiclass
        self.annealing_function_background = annealing_function_background
        self.annealing_function_sampling = annealing_function_sampling
        self.focal_loss_alpha, self.focal_loss_gamma = focal_loss_parameters

    def focal_loss(self, loss,):
        pt = torch.exp(-loss)
        return self.focal_loss_alpha * ((1.0 - pt) ** self.focal_loss_gamma) * loss

    def forward(self, input: Tensor, target: Tensor, epoch: int) -> Tensor:
        """
        Uses the exclusivity threshold and annealing functions as described in the Sparse-shot Learning with Exclusive
        Cross-Entropy for Extremely Many Localisations paper[1] to produce the exclusive cross entropy loss.

        :param input: representing the unlabelled samples, a list of bounding box confidences of shape BxN, or a list
            of pixels from re-shaped images of shape Bx(HxW), or classification predictions of shape B.
            Prediction features should NOT be the output of an activation function such as sigmoid or softmax;
            it is already performed here by the cross entropy.
        :param target: images of shape BxHxW, or a list of bounding box target confidence of shape BxN, or
            classification targets of shape B.
        :return: the exclusive cross entropy loss value
        """
        cross_entropy = super().forward(input, target)

        # Sample
        ece_mask = torch.rand(input.shape).to(torch.device(input.get_device())) < self.annealing_function_sampling(
            epoch, self.sampling_threshold)

        # Get uncertainty score
        if self.multiclass:
            top2classes = torch.topk(input, k=2, dim=1)[0]
            input = torch.abs(top2classes[:, 0, ...] - top2classes[:, 1, ...])

        # Exclude over-confident samples
        ece_mask *= input < self.annealing_function_background(epoch, self.background_threshold)

        # Apply focal re-weighing and sampling
        ece_loss = self.focal_loss(cross_entropy) * ece_mask
        return ece_loss
