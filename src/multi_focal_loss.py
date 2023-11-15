from torchvision.ops.focal_loss import sigmoid_focal_loss
from torch.nn.modules.loss import _Loss

import torch
from torch import Tensor
from typing import Callable, Optional

def multi_label_sigmoid_focal_loss(logits, targets, alphas, gamma=2.0, reduction="mean"):
    """
    Compute the multi-label sigmoid focal loss.
    
    Args:
    - logits (torch.Tensor): raw predictions from the model of shape (batch_size, num_classes).
    - targets (torch.Tensor): true labels of shape (batch_size, num_classes).
    - alphas (torch.Tensor): alpha values for each class of shape (num_classes).
    - gamma (float): focusing parameter.
    - reduction (str): specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 
                       'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements
                       in the output, 'sum': the output will be summed.

    Returns:
    - loss (torch.Tensor): the computed multi-label sigmoid focal loss.
    """
    num_classes = logits.shape[1]
    losses = torch.zeros(logits.shape[0], num_classes, device=logits.device)

    for c in range(num_classes):
        class_logits = logits[:, c]
        class_targets = targets[:, c]
        class_loss = sigmoid_focal_loss(class_logits, class_targets, gamma=gamma, alpha=1.0, reduction='none')
        
        losses[:, c] = class_loss * alphas[c]
    
    if reduction == "none":
        return losses
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "mean":
        return losses.mean()
    else:
        raise ValueError(f"Unsupported reduction mode: {reduction}")

class FocalWithLogitsLoss(_Loss):
    def __init__(
        self, 
        reduction: str = 'mean',
        pos_weight: Optional[Tensor] = None,
        size_average=None, 
        reduce=None,
        ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer('pos_weight', pos_weight)
        self.pos_weight: Optional[Tensor]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return multi_label_sigmoid_focal_loss(
            input, 
            target,
            alphas=self.pos_weight,
            reduction=self.reduction
            )