import importlib
import random

import numpy as np
from scipy.ndimage import rotate, map_coordinates, gaussian_filter, convolve
from skimage import measure
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries

# WARN: use fixed random state for reproducibility; if you want to randomize on each run seed with `time.time()` e.g.
import scipy.ndimage as ndimage


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl, atlas_img=None, atlas_lbl=None):
        for t in self.transforms:
            img, lbl, atlas_img, atlas_lbl = t(img, lbl, atlas_img, atlas_lbl)
        return img, lbl, atlas_img, atlas_lbl


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, axis_prob=0.5, **kwargs):
        self.axes = (2,)
        self.axis_prob = axis_prob

    def __call__(self, img, lbl, atlas_img=None, atlas_lbl=None):
        for axis in self.axes:
            if np.random.uniform(0, 1) < self.axis_prob:
                # img = img[::-1,:,:]
                # lbl = lbl[::-1, :, :]
                # atlas_img = atlas_img[::-1, :, :]
                # atlas_lbl = atlas_lbl[::-1, :, :]
                img = np.flip(img, axis)
                lbl = np.flip(lbl, axis)
                if atlas_img is not None and atlas_lbl is not None:
                    if np.random.uniform(0, 1) < self.axis_prob:
                        atlas_img = np.flip(atlas_img, axis)
                        atlas_lbl = np.flip(atlas_lbl, axis)

        return img, lbl, atlas_img, atlas_lbl


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, angle_spectrum=20, axes=None, mode='reflect', order=3, **kwargs):
        if axes is None:
            axes = ((0, 1), (0, 2), (1, 2))
        else:
            assert isinstance(axes, list) and len(axes) > 0
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, img, lbl, atlas_img=None, atlas_lbl=None):
        for n in range(len(self.axes)):
            angle = np.random.uniform(-self.angle_spectrum, self.angle_spectrum)
            # angle = 5
            img = rotate(img, angle, axes=self.axes[n], reshape=False, order=self.order, mode=self.mode)
            lbl = rotate(lbl, angle, axes=self.axes[n], reshape=False, order=self.order, mode=self.mode)
            lbl[lbl >= 0.5] = 1
            lbl[lbl < 0.5] = 0
            if atlas_img is not None and atlas_lbl is not None:
                angle = np.random.uniform(-self.angle_spectrum, self.angle_spectrum)
                atlas_img = rotate(atlas_img, angle, axes=self.axes[n], reshape=False, order=self.order, mode=self.mode)
                atlas_lbl = rotate(atlas_lbl, angle, axes=self.axes[n], reshape=False, order=self.order, mode=self.mode)
                atlas_lbl[atlas_lbl >= 0.5] = 1
                atlas_lbl[atlas_lbl < 0.5] = 0

        return img, lbl, atlas_img, atlas_lbl


def rot_3d(X, Y, atlas_img=None, atlas_lbl=None, max_angle=40):
    axis = ((0, 1), (0, 2), (1, 2))
    for n in range(len(axis)):
        theta = np.random.uniform(-max_angle, max_angle)
        X = ndimage.rotate(X, theta, axes=axis[n], reshape=False, mode='reflect')
        Y = ndimage.rotate(Y, theta, axes=axis[n], reshape=False, mode='reflect')
        if atlas_img is not None and atlas_lbl is not None:
            theta = np.random.uniform(-max_angle, max_angle)
            atlas_img = ndimage.rotate(atlas_img, theta, axes=axis[n], reshape=False, mode='reflect')
            atlas_lbl = ndimage.rotate(atlas_lbl, theta, axes=axis[n], reshape=False, mode='reflect')
            atlas_lbl = (atlas_lbl > 0.5).astype(int)

        return X, (Y > 0.5).astype(int), atlas_img, atlas_lbl


class RandomContrast:
    """
    Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.
    """

    def __init__(self, alpha=(0.8, 1.2), mean=0.0, execution_probability=0.1, **kwargs):
        assert len(alpha) == 2
        self.alpha = alpha
        self.mean = mean
        self.execution_probability = execution_probability

    def __call__(self, img, lbl, atlas_img=None, atlas_lbl=None):
        if np.random.uniform(0, 1) < self.execution_probability:
            alpha = np.random.uniform(self.alpha[0], self.alpha[1])
            result_img = self.mean + alpha * (img - self.mean)
            img = np.clip(result_img, -1, 1)
            if atlas_img is not None and atlas_lbl is not None:
                alpha = np.random.uniform(self.alpha[0], self.alpha[1])
                result_atlas = self.mean + alpha * (atlas_img - self.mean)
                atlas_img = np.clip(result_atlas, -1, 1)

        return img, lbl, atlas_img, atlas_lbl


# it's relatively slow, i.e. ~1s per patch of size 64x200x200, so use multiple workers in the DataLoader
# remember to use spline_order=0 when transforming the labels
class ElasticDeformation:
    """
    Apply elastic deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, spline_order=3, alpha=(-1500, 1500), sigma=50, execution_probability=0.1, apply_3d=True,
                 **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        :param execution_probability: probability of executing this transform
        :param apply_3d: if True apply deformations in each axis
        """
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability
        self.apply_3d = apply_3d

    def __call__(self, img, lbl, atlas_img=None, atlas_lbl=None):
        if np.random.uniform(0, 1) < self.execution_probability:
            volume_shape = img.shape
            z_alpha = np.random.uniform(self.alpha[0], self.alpha[1])
            y_alpha = np.random.uniform(self.alpha[0], self.alpha[1])
            x_alpha = np.random.uniform(self.alpha[0], self.alpha[1])
            dz = gaussian_filter(np.random.randn(*volume_shape), self.sigma, mode="reflect") * z_alpha
            dy = gaussian_filter(np.random.randn(*volume_shape), self.sigma, mode="reflect") * y_alpha
            dx = gaussian_filter(np.random.randn(*volume_shape), self.sigma, mode="reflect") * x_alpha

            z_dim, y_dim, x_dim = volume_shape
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = z + dz, y + dy, x + dx

            img = map_coordinates(img, indices, order=self.spline_order, mode='reflect')
            lbl = map_coordinates(lbl, indices, order=0, mode='reflect')
            if atlas_img is not None and atlas_lbl is not None:
                # specified parameters for atlas image
                z_alpha = np.random.uniform(self.alpha[0], self.alpha[1])
                y_alpha = np.random.uniform(self.alpha[0], self.alpha[1])
                x_alpha = np.random.uniform(self.alpha[0], self.alpha[1])
                dz = gaussian_filter(np.random.randn(*volume_shape), self.sigma, mode="reflect") * z_alpha
                dy = gaussian_filter(np.random.randn(*volume_shape), self.sigma, mode="reflect") * y_alpha
                dx = gaussian_filter(np.random.randn(*volume_shape), self.sigma, mode="reflect") * x_alpha

                z_dim, y_dim, x_dim = volume_shape
                z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
                indices = z + dz, y + dy, x + dx

                atlas_img = map_coordinates(atlas_img, indices, order=self.spline_order, mode='reflect')
                atlas_lbl = map_coordinates(atlas_lbl, indices, order=0, mode='reflect')

        return img, lbl, atlas_img, atlas_lbl
