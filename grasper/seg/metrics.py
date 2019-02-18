import numpy as np

import torch

class PrecisionRecallScorer(object):
    """
    Score grasping predictions with sparse ground truth using:
    - precision = # correct preds / all preds
    - recall = # correct preds / all true instances of that class
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confmat = np.zeros((self.num_classes * 2, self.num_classes * 2))
        self.scores = {}

    def update(self, output, target, aux):
        # TODO: only works with batch size 1
        output = torch.unbind(output, dim=1)
        output = np.concatenate([(o.data.cpu().numpy() > 0).astype(np.uint8) for o in output], axis=0)
        target = target.data.cpu().numpy()[0]
        # Only consider non-ignore locations
        locations = list(zip(*np.where(target < 255)))
        for (c,x,y) in locations:
            self.confmat[c+4*output[c,x,y], c+4*target[c,x,y]] += 1

    def score(self):
        # overall accuracy
        self.scores['precision'] = sum([np.divide(float(x),y, out=np.zeros_like(x), where=y!=0) for (x,y) in zip(np.diag(self.confmat), np.sum(self.confmat, axis=0))]) \
                / len(np.where(np.sum(self.confmat, axis=1))[0])
        # per-class accuracy
        self.scores['recall'] = sum([np.divide(float(x), y, out=np.zeros_like(x), where=y!=0) for (x,y) in zip(np.diag(self.confmat), np.sum(self.confmat, axis=1))]) \
                / len(np.where(np.sum(self.confmat, axis=1))[0])
        return self.scores

    def save(self, path):
        np.savez(path, scores=self.scores)
