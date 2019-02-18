import torch

from .fcn32s import fcn32s
from .fcn32s_patch import fcn32s_patch

models = {
    'fcn32s': fcn32s,
    'fcn32s-patch': fcn32s_patch,
}

def prepare_model(model_name, num_classes, weights=None):
    model = models[model_name](num_classes)
    if weights:
        model.load_state_dict(weights)
    return model
