from .fcn32s import fcn32s
import torch.nn.functional as F

import pdb

class fcn32s_patch(fcn32s):
    """
    FCN-32s trained with patch sampling
    """

    def __init__(self, num_classes, feat_dim=None):
        super().__init__(num_classes, feat_dim)
        # pad differs between train and eval, so pad dynamically
        #self.encoder[0].padding = (1, 1)   # VGG

    def forward(self, x, training=False):
        h, w = x.size()[-2:]
        # inference is on the whole image
        if not training:
            #x = F.pad(x, (80, 80, 80, 80), 'constant', 0)  # VGG
            x = F.pad(x, (20, 20, 20, 20), 'constant', 0)   # RESNET
            x = self.encoder(x)
            x = self.head(x)
            x = self.decoder(x)
            x = x[..., self.crop:self.crop + h, self.crop:self.crop + w]
        # train on patches
        else:
            x = self.encoder(x)
            x = self.head(x)
        return x
