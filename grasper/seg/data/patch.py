import sys
import numpy as np

from torch.utils.data import Dataset

from .util import Wrapper
from util import utils

import pdb

class PatchSeg(Wrapper, Dataset):
    """
    Sample a receptive field (404 x 404) pixel patch from image / label
    pairs loaded from the wrapped dataset
    """
    # negatives parameters makes a dataset that retrieves the negative patch
    def __init__(self, dataset, negatives=False):
        super().__init__(dataset)
        self.ds = dataset
        self.negatives = negatives

    def __getitem__(self, idx):
        im, target, aux = self.ds[idx]
        if 'val' in self.split:
            return im, target, aux
        else:
            # pad image and target to allow a crop centered at any pixel in the image
            ms = 202
            pad_target = np.pad(target, ((ms, ms), (ms, ms)), 'constant', constant_values=255)
            if self.negatives:
                loc = utils.mj_to_img(aux['other_loc'][0]).astype(int) + np.array([ms,ms])
                aux['negative'] = True
            else:
                loc = np.unravel_index(np.random.choice(np.where(pad_target.flat != 255)[0]), pad_target.shape)
            x, y = np.array(im).shape[:2]
            pad_im = np.full((x + ms*2, y + ms*2, 3), np.array(self.mean)*255., dtype=np.uint8)
            pad_im[ms:-ms, ms:-ms, :] = im
            patch_im = pad_im[loc[0] - ms: loc[0] + ms, loc[1] - ms: loc[1] + ms, :]
            # construct patch label
            patch_lbl = np.full((7,7), 255, dtype=np.uint8)
            patch_lbl[3,3] = pad_target[loc[0], loc[1]]
            return patch_im, patch_lbl, aux

    def __len__(self):
        return len(self.slugs)
