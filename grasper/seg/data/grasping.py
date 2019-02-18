import sys
import numpy as np
import random
import h5py
import math
from abc import abstractmethod
from pathlib import Path
import PIL
from PIL import Image

from torch.utils.data import Dataset

from .util import Wrapper
from util import utils

import matplotlib.pyplot as plt
import pdb

class GraspingData(Dataset):
    """
    Skeleton for loading grasping data with the form
    (image, grasp parameters, success)
    into image-target pairs for a fully convolutional grasp predictor
    """
    classes = None

    # pixel stats (RGB)
    mean = (0., 0., 0.)
    std = (1., 1., 1.)

    # exclude this target value from the loss
    ignore_index = None

    def __init__(self, root_dir=None, split=None):
        self.root_dir = Path(root_dir)
        self.split = split

        self.slugs = self.load_slugs()

    @abstractmethod
    def load_slugs(self):
        pass

    @abstractmethod
    def slug_to_data_path(self, slug):
        pass

    def load_data(self, path):
        return h5py.File(path, 'r')

    def load_image(self, data):
        return np.array(data.get('image'), dtype=np.uint8)

    def load_annotation(self, data):
        # convert grasp params and success information into image label
        grasp_loc = np.array(data.get('grasp_loc'))[0]
        end_state = np.array(data.get('end').value)
        init_state = np.array(data.get('image').value)
        other_loc = np.array(data.get('other_loc'))
        grasp_angle = np.array(data.get('grasp_angle'))[0]
        is_gripping = np.array(data.get('is_gripping'))[0]
        sx, sy, _ = np.array(data.get('image')).shape

        def mj_to_img(p):
            # Black magic, gotten through least squares
            # Most likely won't work for images that aren't 500x500
            A = np.array([1570.87903103, -1570.54846599])
            b = np.array([248.25918547, 249.76304022])

            return A * p + b

        def bin_angle(a):
            # disccretize angle into (0, 45, 90, 135) degrees
            a += math.pi / 2 # range [-pi/2, pi/2] -> [0, pi]
            if a < math.pi / 2:
                cl = 0 if a < math.pi / 4 else 1
            else:
                cl = 2 if a < 3*math.pi / 4 else 3
            return cl

        label = np.full((sx, sy), self.ignore_index, dtype=np.uint8)

        # Flip x y for numpy
        y, x = mj_to_img(grasp_loc)
        x = int(x)
        y = int(y)

        cl = bin_angle(grasp_angle)
        # TODO this is a little hacky - trying to push difference btwn
        # softmax and sigCE targets to transforms

        # [0-self.num_classes) -> gripped
        # [self.num_classes-2*self.num_classes) -> not gripped
        label[x, y] = cl + self.num_classes if is_gripping else cl

        # CHANGE
        return label, ((y, x), other_loc, end_state, init_state, grasp_angle, is_gripping)

    @property
    def num_classes(self):
        return len(self.classes)

    def __getitem__(self, idx):
        slug = idx
        if isinstance(idx, int):
            slug = self.slugs[idx]
        data = self.load_data(self.slug_to_data_path(slug))
        im = self.load_image(data)
        target = self.load_annotation(data)
        # third return is reserved for auxiliary info dict
        return im, target, {'idx': slug}

    def __len__(self):
        return len(self.slugs)


class MJGrasp(GraspingData):
    """
    Load Mujoco simulated grasping data

    Args:
        root_dir: path to simulated data
        split: {train, val}
    """

    mean = (0.612, 0.5169, 0.4455)
    classes = ['0', '135', '90', '45']

    ignore_index = 255

    def __init__(self, rotate_augment=False, **kwargs):
        kwargs['root_dir'] = kwargs.get('root_dir', 'data/grasp_sim')
        kwargs['split'] = kwargs.get('split', 'train')
        self.rotate = rotate_augment
        super().__init__(**kwargs)

    def load_slugs(self):
        listing = self.listing_path()
        with open(listing, 'r') as f:
            grasps = f.read().splitlines()
        # augment each data point with rotated versions
        slugs = []
        for g in grasps:
            num_rot = len(self.classes) if self.split == 'train' else 1
            slugs += [(g, x) for x in range(num_rot)]
        return slugs

    def listing_path(self):
        return str(self.root_dir / '{}.txt'.format(self.split))

    def slug_to_data_path(self, slug):
        return str(self.root_dir / 'train' / '{}.hdf5'.format(slug[0]))

    def get_rotation_matrix(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        return R

    def rotate_label(self, im, rotation):
        # rotate target with single labeled point manually
        # first compute the correct class label
        cl = im.flat[np.where(im.flat != 255)[0]]
        is_gripping = (cl >= self.num_classes)
        cl = (cl + rotation) % 4
        cl = cl + self.num_classes if is_gripping else cl
        # first pad image so we don't rotate outide of it
        diag = math.ceil(np.sqrt(np.square(im.shape[0]) + np.square(im.shape[1])))
        hp, vp = math.ceil((diag - im.shape[0]) / 2), math.ceil((diag - im.shape[1]) / 2)
        im = np.pad(im, ((hp, hp), (vp, vp)), 'constant', constant_values=self.ignore_index)
        # center grid at center of image
        center = [im.shape[0] // 2, im.shape[1] //2]
        # get vector to labeled point
        pt = list(zip(*np.where(im != self.ignore_index)))[0]
        pt = np.array([p - c for p, c in zip(pt, center)])
        # rotate vector
        angle = (int(self.classes[rotation]) / 360) *2*np.pi
        rot_matrix = self.get_rotation_matrix(angle)
        pt = (rot_matrix.dot(pt)).astype(np.int)
        # make a target with label at this new coordinate
        label = np.full_like(im, self.ignore_index)
        label[pt[0] + center[0], pt[1] + center[1]] = cl
        return label

    def rotate_im(self, im, rotation):
        # rotate image by given rotation amount
        angle = int(self.classes[rotation])
        old_shape = im.shape
        im = Image.fromarray(im.astype(np.uint8))
        im = im.rotate(angle, resample=PIL.Image.BILINEAR, expand=1)
        im = np.array(im, dtype=np.uint8)
        # pad if needed so all rotations are the same size
        if old_shape == im.shape:
            diag = math.ceil(np.sqrt(np.square(im.shape[0]) + np.square(im.shape[1])))
            hp, vp = math.ceil((diag - im.shape[0]) / 2), math.ceil((diag - im.shape[1]) / 2)
            im = np.pad(im, ((hp, hp), (vp, vp), (0,0)), 'constant', constant_values=0)
        # set default fill to data mean
        mean_im = np.full_like(im, np.array(self.mean)*255., dtype=np.uint8)
        mean_im.flat[im.flat != 0] = im.flat[im.flat != 0]
        im = mean_im
        return im

    def __getitem__(self, idx):
        slug = idx
        if isinstance(idx, int):
            slug = self.slugs[idx]
        data = self.load_data(self.slug_to_data_path(slug))

        if self.rotate:
            im = self.rotate_im(self.load_image(data), slug[1])
        else:
            im = self.load_image(data)

        annotation, (grasp_loc, other_loc, end_state, init_state, grasp_angle, is_gripping) = self.load_annotation(data)

        if self.rotate:
            target = self.rotate_label(annotation, slug[1])
        else:
            target = annotation

        aux = {'idx': slug[0], 'other_loc': other_loc, 'end_state': end_state, 'init_state': init_state, 'grasp_loc': grasp_loc, 'grasp_angle': grasp_angle, 'is_gripping': str(is_gripping)}
        #pdb.set_trace()

        # third return is reserved for auxiliary info dict
        return im, target, aux


