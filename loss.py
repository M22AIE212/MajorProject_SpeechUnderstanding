import torch.nn as nn
import torch.nn.functional as F

class CustomCrossEntropyLoss(nn.Module):
    """
    Custom implementation of Cross Entropy Loss.

    Args:
    - None
    """

    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, model_output, target_label):
        target_label = target_label.long()
        loss = F.cross_entropy(model_output, target_label)
        return loss
