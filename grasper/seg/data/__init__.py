import os
import pickle

import torch
from torchvision.transforms import Compose

from .patch import PatchSeg
from .grasping import MJGrasp
from .transforms import *
from .util import InputsTargetAuxCollate

import pdb

def prepare_grasping_patch_data(dataset_name, split, root_dir='grasp_sim', negative=False):
    dataset = datasets[dataset_name]
    ds = dataset(split=split, root_dir=root_dir)
    #ds = dataset(split=split)
    patch_ds = PatchSeg(ds, negative)
    image_transform = Compose([
        ImToCaffe(mean=dataset.mean),
        NpToTensor()
    ])
    target_transform = Compose([ExplodeTarget(ds.num_classes), SegToTensor()])
    #transform_ds = TransformData(patch_ds, input_transforms=[image_transform],
        #target_transform=target_transform)
    transform_ds = TransformData(ds, input_transforms=[image_transform],
        target_transform=target_transform)
    return transform_ds

def prepare_loader(dataset, batch=1, evaluation=False):
    shuffle = True
    num_workers = 1
    if evaluation:
        shuffle = False
        num_workers = 0  # for determinism
    return torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True)
image_transform = Compose([
    ImToCaffe(mean= (0.612, 0.5169, 0.4455)),
    NpToTensor()
])
def transform_image(image):
    return image_transform(image)

datasets = {
    'mj-grasping': MJGrasp,
}

datatypes = {
    'grasping-patch': prepare_grasping_patch_data,
}

