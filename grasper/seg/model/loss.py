import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class CrossEntropyLoss2D(nn.CrossEntropyLoss):
    """
    Extend softmax + CE loss module to compute loss over spatial (2D) inputs.

    Take
        scores: the predictions with shape N x C x H x W
        target: the true target with shape N x 1 x H x W
    """

    def forward(self, scores, target):
        if len(scores.size()) != 4:
            raise ValueError("Scores should have 4 dimensions, but has {}: {}".format(len(scores.size()), scores.size()))
        _, c, _, _ = scores.size()
        scores = scores.permute(0, 2, 3, 1).contiguous().view(-1, c)
        target = target.view(-1)
        return F.cross_entropy(scores, target, self.weight, self.size_average,
                               self.ignore_index, self.reduce)


class SigmoidCrossEntropyLoss2D(_Loss):
    '''
    Extend sigmoid + CE loss to compute loss over spatial (2D) inputs
    and mask the loss for ignored targets.

    Take
        scores: N x C x H x W
        target: N x C x H x W
    '''

    def __init__(self, size_average=True, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, scores, target):
        # make mask to exclude ignore index
        mask = target.data.new(*target.size()).byte()
        mask[...] = (target.data != self.ignore_index)
        # mask scores and targets accordingly
        scores = scores[mask]
        target = target[mask]
        # sigmoid + cross entropy loss
        target = target.float()
        return F.binary_cross_entropy_with_logits(scores, target)
