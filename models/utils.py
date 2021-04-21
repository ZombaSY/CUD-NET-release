import numpy as np
import PIL.Image as Image
import torch
import random
import colorsys
import math
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import torchvision

from torch.autograd import Variable
from PIL.ImageOps import invert
from torchvision.utils import make_grid
from torchvision import transforms
from typing import Union, Optional, List, Tuple, Text, BinaryIO
from skimage import io, color
from skimage.measure import compare_ssim as ssim
from matplotlib.image import imread
from matplotlib import pyplot as plt


class ImageProcessing(object):
    '''
    @issue
    'hsv_to_rgb' and 'rgb_to_hsv' convert the image with H 180 value to 0, resulting blue color to red color

    '''

    @staticmethod
    def rgb_to_lab(img, is_training=True):
        """ PyTorch implementation of RGB to LAB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
        :param img: image to be adjusted
        :returns: adjusted image
        :rtype: Tensor

        """
        img = img.permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()
        img = img.view(-1, 3)

        img = (img / 12.92) * img.le(0.04045).float() + (((torch.clamp(img,
                                                                       min=0.0001) + 0.055) / 1.055) ** 2.4) * img.gt(
            0.04045).float()

        rgb_to_xyz = Variable(torch.FloatTensor([  # X        Y          Z
            [0.412453, 0.212671, 0.019334],  # R
            [0.357580, 0.715160, 0.119193],  # G
            [0.180423, 0.072169,
             0.950227],  # B
        ]), requires_grad=False).cuda()

        img = torch.matmul(img, rgb_to_xyz)
        img = torch.mul(img, Variable(torch.FloatTensor(
            [1 / 0.950456, 1.0, 1 / 1.088754]), requires_grad=False).cuda())

        epsilon = 6 / 29

        img = ((img / (3.0 * epsilon ** 2) + 4.0 / 29.0) * img.le(epsilon ** 3).float()) + \
              (torch.clamp(img, min=0.0001) **
               (1.0 / 3.0) * img.gt(epsilon ** 3).float())

        fxfyfz_to_lab = Variable(torch.FloatTensor([[0.0, 500.0, 0.0],  # fx
                                                    # fy
                                                    [116.0, -500.0, 200.0],
                                                    # fz
                                                    [0.0, 0.0, -200.0],
                                                    ]), requires_grad=False).cuda()

        img = torch.matmul(img, fxfyfz_to_lab) + Variable(
            torch.FloatTensor([-16.0, 0.0, 0.0]), requires_grad=False).cuda()

        img = img.view(shape)
        img = img.permute(2, 1, 0)

        '''
        L_chan: black and white with input range [0, 100]
        a_chan/b_chan: color channels with input range ~[-110, 110], not exact 
        [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
        '''
        img[0, :, :] = img[0, :, :] / 100
        img[1, :, :] = (img[1, :, :] / 110 + 1) / 2
        img[2, :, :] = (img[2, :, :] / 110 + 1) / 2

        img[(img != img).detach()] = 0

        img = img.contiguous()

        return img.cuda()

    @staticmethod
    def lab_to_rgb(img, is_training=True):
        """ PyTorch implementation of LAB to RGB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
        :param img: image to be adjusted
        :returns: adjusted image
        :rtype: Tensor
        """
        img = img.permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()
        img = img.view(-1, 3)
        img_copy = img.clone()

        img_copy[:, 0] = img[:, 0] * 100
        img_copy[:, 1] = ((img[:, 1] * 2) - 1) * 110
        img_copy[:, 2] = ((img[:, 2] * 2) - 1) * 110

        img = img_copy.clone().cuda()
        del img_copy

        lab_to_fxfyfz = Variable(torch.FloatTensor([  # X Y Z
            [1 / 116.0, 1 / 116.0, 1 / 116.0],  # R
            [1 / 500.0, 0, 0],  # G
            [0, 0, -1 / 200.0],  # B
        ]), requires_grad=False).cuda()

        img = torch.matmul(
            img + Variable(torch.cuda.FloatTensor([16.0, 0.0, 0.0])), lab_to_fxfyfz)

        epsilon = 6.0 / 29.0

        img = (((3.0 * epsilon ** 2 * (img - 4.0 / 29.0)) * img.le(epsilon).float()) +
               ((torch.clamp(img, min=0.0001) ** 3.0) * img.gt(epsilon).float()))

        # denormalize for D65 white point
        img = torch.mul(img, Variable(
            torch.cuda.FloatTensor([0.950456, 1.0, 1.088754])))

        xyz_to_rgb = Variable(torch.FloatTensor([  # X Y Z
            [3.2404542, -0.9692660, 0.0556434],  # R
            [-1.5371385, 1.8760108, -0.2040259],  # G
            [-0.4985314, 0.0415560, 1.0572252],  # B
        ]), requires_grad=False).cuda()

        img = torch.matmul(img, xyz_to_rgb)

        img = (img * 12.92 * img.le(0.0031308).float()) + ((torch.clamp(img,
                                                                        min=0.0001) ** (
                                                                        1 / 2.4) * 1.055) - 0.055) * img.gt(
            0.0031308).float()

        img = img.view(shape)
        img = img.permute(2, 1, 0)

        img = img.contiguous()
        img[(img != img).detach()] = 0

        return img

    @staticmethod
    def swapimdims_3HW_HW3(img):
        """Move the image channels to the first dimension of the numpy
        multi-dimensional array

        :param img: numpy nd array representing the image
        :returns: numpy nd array with permuted axes
        :rtype: numpy nd array

        """
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 1, 2), 0, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 2, 3), 1, 3)

    @staticmethod
    def swapimdims_HW3_3HW(img):
        """Move the image channels to the last dimensiion of the numpy
        multi-dimensional array

        :param img: numpy nd array representing the image
        :returns: numpy nd array with permuted axes
        :rtype: numpy nd array

        """
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 1, 3), 2, 3)

    @staticmethod
    def load_image(img_filepath, normaliser):
        """Loads an image from file as a numpy multi-dimensional array

        :param img_filepath: filepath to the image
        :returns: image as a multi-dimensional numpy array
        :rtype: multi-dimensional numpy array

        """
        img = ImageProcessing.normalise_image(
            imread(img_filepath), normaliser)  # NB: imread normalises to 0-1
        return img

    @staticmethod
    def normalise_image(img, normaliser):
        """Normalises image data to be a float between 0 and 1

        :param img: Image as a numpy multi-dimensional image array
        :returns: Normalised image as a numpy multi-dimensional image array
        :rtype: Numpy array

        """
        img = img.astype('float32') / normaliser
        return img

    @staticmethod
    def compute_mse(original, result):
        """Computes the mean squared error between to RGB images represented as multi-dimensional numpy arrays.

        :param original: input RGB image as a numpy array
        :param result: target RGB image as a numpy array
        :returns: the mean squared error between the input and target images
        :rtype: float

        """
        return ((original - result) ** 2).mean()

    @staticmethod
    def compute_psnr(image_batchA, image_batchB, max_intensity):
        """Computes the PSNR for a batch of input and output images

        :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param max_intensity: maximum intensity possible in the image (e.g. 255)
        :returns: average PSNR for the batch of images
        :rtype: float

        """
        num_images = image_batchA.shape[0]
        psnr_val = 0.0

        for i in range(0, num_images):
            imageA = image_batchA[i, 0:3, :, :]
            imageB = image_batchB[i, 0:3, :, :]
            imageB = np.maximum(0, np.minimum(imageB, max_intensity))
            psnr_val += 10 * \
                        np.log10(max_intensity ** 2 /
                                 ImageProcessing.compute_mse(imageA, imageB))

        return psnr_val / num_images

    @staticmethod
    def compute_ssim(image_batchA, image_batchB):
        """Computes the SSIM for a batch of input and output images

        :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param max_intensity: maximum intensity possible in the image (e.g. 255)
        :returns: average PSNR for the batch of images
        :rtype: float

        """
        num_images = image_batchA.shape[0]
        ssim_val = 0.0

        for i in range(0, num_images):
            imageA = ImageProcessing.swapimdims_3HW_HW3(
                image_batchA[i, 0:3, :, :])
            imageB = ImageProcessing.swapimdims_3HW_HW3(
                image_batchB[i, 0:3, :, :])
            ssim_val += ssim(imageA, imageB, data_range=imageA.max() - imageA.min(), multichannel=True,
                             gaussian_weights=True, win_size=11)

        return ssim_val / num_images

    @staticmethod
    def hsv_to_rgb(img):
        """Converts a HSV image to RGB
        PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

        :param img: HSV image
        :returns: RGB image
        :rtype: Tensor

        """
        img = torch.clamp(img, 0, 1)
        img = img.permute(2, 1, 0)

        m1 = 0
        m2 = (img[:, :, 2] * (1 - img[:, :, 1]) - img[:, :, 2]) / 60
        m3 = 0
        m4 = -1 * m2
        m5 = 0

        r = img[:, :, 2] + torch.clamp(img[:, :, 0] * 360 - 0, 0, 60) * m1 + torch.clamp(img[:, :, 0] * 360 - 60, 0,
                                                                                         60) * m2 + torch.clamp(
            img[:, :, 0] * 360 - 120, 0, 120) * m3 + torch.clamp(img[:, :, 0] * 360 - 240, 0, 60) * m4 + torch.clamp(
            img[:, :, 0] * 360 - 300, 0, 60) * m5

        m1 = (img[:, :, 2] - img[:, :, 2] * (1 - img[:, :, 1])) / 60
        m2 = 0
        m3 = -1 * m1
        m4 = 0

        g = img[:, :, 2] * (1 - img[:, :, 1]) + torch.clamp(img[:, :, 0] * 360 - 0, 0, 60) * m1 + torch.clamp(
            img[:, :, 0] * 360 - 60,
            0, 120) * m2 + torch.clamp(img[:, :, 0] * 360 - 180, 0, 60) * m3 + torch.clamp(img[:, :, 0] * 360 - 240, 0,
                                                                                           120) * m4

        m1 = 0
        m2 = (img[:, :, 2] - img[:, :, 2] * (1 - img[:, :, 1])) / 60
        m3 = 0
        m4 = -1 * m2

        b = img[:, :, 2] * (1 - img[:, :, 1]) + torch.clamp(img[:, :, 0] * 360 - 0, 0, 120) * m1 + torch.clamp(
            img[:, :, 0] * 360 -
            120, 0, 60) * m2 + torch.clamp(img[:, :, 0] * 360 - 180, 0, 120) * m3 + torch.clamp(
            img[:, :, 0] * 360 - 300, 0, 60) * m4

        img = torch.stack((r, g, b), 2)
        img[(img != img).detach()] = 0

        img = img.permute(2, 1, 0)
        img = img.contiguous()
        img = torch.clamp(img, 0, 1)

        return img

    @staticmethod
    def rgb_to_hsv(img):
        """Converts an RGB image to HSV
        PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

        :param img: RGB image
        :returns: HSV image
        :rtype: Tensor

        """
        img = torch.clamp(img, 0.000000001, 1)

        img = img.permute(2, 1, 0)
        # 3, H, W
        shape = img.shape

        img = img.contiguous()
        img = img.view(-1, 3)

        mx = torch.max(img, 1)[0]
        mn = torch.min(img, 1)[0]

        ones = Variable(torch.FloatTensor(
            torch.ones((img.shape[0])))).cuda()
        zero = Variable(torch.FloatTensor(torch.zeros(shape[0:2]))).cuda()

        img = img.view(shape)

        ones1 = ones[0:math.floor((ones.shape[0] / 2))]
        ones2 = ones[math.floor(ones.shape[0] / 2):(ones.shape[0])]

        mx1 = mx[0:math.floor((ones.shape[0] / 2))]
        mx2 = mx[math.floor(ones.shape[0] / 2):(ones.shape[0])]
        mn1 = mn[0:math.floor((ones.shape[0] / 2))]
        mn2 = mn[math.floor(ones.shape[0] / 2):(ones.shape[0])]

        df1 = torch.add(mx1, torch.mul(ones1 * -1, mn1))
        df2 = torch.add(mx2, torch.mul(ones2 * -1, mn2))

        df = torch.cat((df1, df2), 0)
        del df1, df2
        df = df.view(shape[0:2]) + 1e-10
        mx = mx.view(shape[0:2])

        img = img.cuda()
        df = df.cuda()
        mx = mx.cuda()

        g = img[:, :, 1].clone().cuda()
        b = img[:, :, 2].clone().cuda()
        r = img[:, :, 0].clone().cuda()

        img_copy = img.clone()

        img_copy[:, :, 0] = (((g - b) / df) * r.eq(mx).float() + (2.0 + (b - r) / df)
                             * g.eq(mx).float() + (4.0 + (r - g) / df) * b.eq(mx).float())
        img_copy[:, :, 0] = img_copy[:, :, 0] * 60.0

        zero = zero.cuda()
        img_copy2 = img_copy.clone()

        img_copy2[:, :, 0] = img_copy[:, :, 0].lt(zero).float(
        ) * (img_copy[:, :, 0] + 360) + img_copy[:, :, 0].ge(zero).float() * (img_copy[:, :, 0])

        img_copy2[:, :, 0] = img_copy2[:, :, 0] / 360

        del img, r, g, b

        img_copy2[:, :, 1] = mx.ne(zero).float() * (df / mx) + \
                             mx.eq(zero).float() * (zero)
        img_copy2[:, :, 2] = mx

        img_copy2[(img_copy2 != img_copy2).detach()] = 0

        img = img_copy2.clone()

        img = img.permute(2, 1, 0)
        img = torch.clamp(img, 0.000000001, 1)

        return img

    @staticmethod
    def apply_curve(img, C, slope_sqr_diff, channel_in, channel_out,
                    clamp=True, same_channel=True):
        """Applies a peicewise linear curve defined by a set of knot points to
        an image channel

        :param img: image to be adjusted
        :param C: predicted knot points of curve
        :returns: adjusted image
        :rtype: Tensor

        """
        slope = Variable(torch.zeros((C.shape[0] - 1))).cuda()
        curve_steps = C.shape[0] - 1
        '''
        Compute the slope of the line segments
        '''
        for i in range(0, C.shape[0] - 1):
            slope[i] = C[i + 1] - C[i]

        '''
        Compute the squared difference between slopes
        '''
        for i in range(0, slope.shape[0] - 1):
            slope_sqr_diff += (slope[i + 1] - slope[i]) * (slope[i + 1] - slope[i])

        '''
        Use predicted line segments to compute scaling factors for the channel
        '''
        scale = float(C[0])
        for i in range(0, slope.shape[0] - 1):
            if clamp:
                scale += float(slope[i]) * (torch.clamp(img[:, :, channel_in] * curve_steps - i, 0, 1))
                # scale += float(slope[i]) * (torch.clamp(img[:, :, channel_in], 0, 1))
            else:
                scale += float(slope[i]) * (img[:, :, channel_in] * curve_steps - i)
        img_copy = img.clone()

        if same_channel:
            # channel in and channel out are the same channel
            img_copy[:, :, channel_out] = img[:, :, channel_in] * scale
        else:
            # otherwise
            img_copy[:, :, channel_out] = img[:, :, channel_out] * scale

        img_copy = torch.clamp(img_copy, 0, 1)

        return img_copy, slope_sqr_diff

    @staticmethod
    def adjust_hsv(img, S):
        """Adjust the HSV channels of a HSV image using learnt curves

        :param img: image to be adjusted
        :param S: predicted parameters of piecewise linear curves
        :returns: adjust image, regularisation term
        :rtype: Tensor, float

        """
        img = img.squeeze(0).permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()

        S1 = torch.exp(S[0:int(S.shape[0] / 4)])
        S2 = torch.exp(S[(int(S.shape[0] / 4)):(int(S.shape[0] / 4) * 2)])
        S3 = torch.exp(S[(int(S.shape[0] / 4) * 2):(int(S.shape[0] / 4) * 3)])
        S4 = torch.exp(S[(int(S.shape[0] / 4) * 3):(int(S.shape[0] / 4) * 4)])

        slope_sqr_diff = Variable(torch.zeros(1) * 0.0).cuda()

        '''
        Adjust Hue channel based on Hue using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, S1, slope_sqr_diff, channel_in=0, channel_out=0)

        '''
        Adjust Saturation channel based on Hue using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S2, slope_sqr_diff, channel_in=0, channel_out=1, same_channel=False)

        '''
        Adjust Saturation channel based on Saturation using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S3, slope_sqr_diff, channel_in=1, channel_out=1)

        '''
        Adjust Value channel based on Value using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S4, slope_sqr_diff, channel_in=2, channel_out=2)

        img = img_copy.clone()
        del img_copy

        img[(img != img).detach()] = 0

        img = img.permute(2, 1, 0)
        img = img.contiguous()

        return img, slope_sqr_diff

    @staticmethod
    def adjust_sv(img, S):
        """Adjust the HSV channels of a HSV image using learnt curves

        :param img: image to be adjusted
        :param S: predicted parameters of piecewise linear curves
        :returns: adjust image, regularisation term
        :rtype: Tensor, float

        """
        img = img.squeeze(0).permute(2, 1, 0)
        img = img.contiguous()

        S3 = torch.exp(S[(int(S.shape[0] / 2) * 0):(int(S.shape[0] / 2) * 1)])
        S4 = torch.exp(S[(int(S.shape[0] / 2) * 1):(int(S.shape[0] / 2) * 2)])

        slope_sqr_diff = Variable(torch.zeros(1) * 0.0).cuda()

        '''
        Adjust Saturation channel based on Saturation using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, S3, slope_sqr_diff, channel_in=1, channel_out=1)

        '''
        Adjust Value channel based on Value using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S4, slope_sqr_diff, channel_in=2, channel_out=2)

        img = img_copy.clone()
        del img_copy

        img[(img != img).detach()] = 0

        img = img.permute(2, 1, 0)
        img = img.contiguous()

        return img, slope_sqr_diff

    @staticmethod
    def adjust_rgb(img, R):
        """Adjust the RGB channels of a RGB image using learnt curves

        :param img: image to be adjusted
        :param S: predicted parameters of piecewise linear curves
        :returns: adjust image, regularisation term
        :rtype: Tensor, float

        """
        img = img.squeeze(0).permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()

        '''
        Extract the parameters of the three curves
        '''
        R1 = torch.exp(R[0:int(R.shape[0] / 3)])
        R2 = torch.exp(R[(int(R.shape[0] / 3)):(int(R.shape[0] / 3) * 2)])
        R3 = torch.exp(R[(int(R.shape[0] / 3) * 2):(int(R.shape[0] / 3) * 3)])

        '''
        Apply the curve to the R channel 
        '''
        slope_sqr_diff = Variable(torch.zeros(1) * 0.0).cuda()

        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, R1, slope_sqr_diff, channel_in=0, channel_out=0)

        '''
        Apply the curve to the G channel 
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, R2, slope_sqr_diff, channel_in=1, channel_out=1)

        '''
        Apply the curve to the B channel 
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, R3, slope_sqr_diff, channel_in=2, channel_out=2)

        img = img_copy.clone()
        del img_copy

        img[(img != img).detach()] = 0

        img = img.permute(2, 1, 0)
        img = img.contiguous()

        return img, slope_sqr_diff

    @staticmethod
    def adjust_lab(img, L):
        """Adjusts the image in LAB space using the predicted curves

        :param img: Image tensor
        :param L: Predicited curve parameters for LAB channels
        :returns: adjust image, and regularisation parameter
        :rtype: Tensor, float

        """
        img = img.permute(2, 1, 0)

        shape = img.shape
        img = img.contiguous()

        '''
        Extract predicted parameters for each L,a,b curve
        '''
        L1 = torch.exp(L[0:int(L.shape[0] / 3)])
        L2 = torch.exp(L[(int(L.shape[0] / 3)):(int(L.shape[0] / 3) * 2)])
        L3 = torch.exp(L[(int(L.shape[0] / 3) * 2):(int(L.shape[0] / 3) * 3)])

        slope_sqr_diff = Variable(torch.zeros(1) * 0.0).cuda()

        '''
        Apply the curve to the L channel 
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, L1, slope_sqr_diff, channel_in=0, channel_out=0)

        '''
        Now do the same for the a channel
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, L2, slope_sqr_diff, channel_in=1, channel_out=1)

        '''
        Now do the same for the b channel
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, L3, slope_sqr_diff, channel_in=2, channel_out=2)

        img = img_copy.clone()
        del img_copy

        img[(img != img).detach()] = 0

        img = img.permute(2, 1, 0)
        img = img.contiguous()

        return img, slope_sqr_diff


class Datonize():
    """
    Code adapted: https://github.com/joergdietrich/daltonize
    """

    @staticmethod
    def __transform_colorspace(img, mat):
        """Transform image to a different color space.

        Arguments:
        ----------
        img : array of shape (M, N, 3)
        mat : array of shape (3, 3)
            conversion matrix to different color space

        Returns:
        --------
        out : array of shape (M, N, 3)
        """
        # Fast element (=pixel) wise matrix multiplication
        return np.einsum("ij, ...j", mat, img, dtype=np.float16, casting="same_kind")

    @staticmethod
    def __simulate(rgb, color_deficit="d"):
        """Simulate the effect of color blindness on an image.

        Arguments:
        ----------
        rgb : array of shape (M, N, 3)
            original image in RGB format
        color_deficit : {"d", "p", "t"}, optional
            type of colorblindness, d for deuteronopia (default),
            p for protonapia,
            t for tritanopia

        Returns:
        --------
        sim_rgb : array of shape (M, N, 3)
            simulated image in RGB format
        """
        # Colorspace transformation matrices
        cb_matrices = {
            "d": np.array([[1, 0, 0], [1.10104433, 0, -0.00901975], [0, 0, 1]], dtype=np.float16),
            "p": np.array([[0, 0.90822864, 0.008192], [0, 1, 0], [0, 0, 1]], dtype=np.float16),
            "t": np.array([[1, 0, 0], [0, 1, 0], [-0.15773032, 1.19465634, 0]], dtype=np.float16),
        }
        rgb2lms = np.array([[0.3904725, 0.54990437, 0.00890159],
                            [0.07092586, 0.96310739, 0.00135809],
                            [0.02314268, 0.12801221, 0.93605194]], dtype=np.float16)
        # Precomputed inverse
        lms2rgb = np.array([[2.85831110e+00, -1.62870796e+00, -2.48186967e-02],
                            [-2.10434776e-01, 1.15841493e+00, 3.20463334e-04],
                            [-4.18895045e-02, -1.18154333e-01, 1.06888657e+00]], dtype=np.float16)
        # first go from RBG to LMS space
        lms = Datonize.__transform_colorspace(rgb, rgb2lms)
        # Calculate image as seen by the color blind
        sim_lms = Datonize.__transform_colorspace(lms, cb_matrices[color_deficit])
        # Transform back to RBG
        sim_rgb = Datonize.__transform_colorspace(sim_lms, lms2rgb)

        return sim_rgb

    @staticmethod
    def __array_to_img(arr, gamma=2.4):
        """Convert a numpy array to a PIL image.

        Arguments:
        ----------
        arr : array of shape (M, N, 3)
        gamma : float exponent of gamma correction

        Returns:
        --------
        img : PIL.Image.Image
            RGB image created from array
        """
        # clip values to lie in the range [0, 255]
        arr = Datonize.__inverse_gamma_correction(arr, gamma=gamma)
        arr = Datonize.__clip_array(arr)
        arr = arr.astype('uint8')
        img = Image.fromarray(arr, mode='RGB')
        return img

    @staticmethod
    def __clip_array(arr, min_value=0, max_value=255):
        """Ensure that all values in an array are between min and max values.

        Arguments:
        ----------
        arr : array_like
        min_value : float, optional
            default 0
        max_value : float, optional
            default 255

        Returns:
        --------
        arr : array_like
            clipped such that all values are min_value <= arr <= max_value
        """
        comp_arr = np.ones_like(arr)
        arr = np.maximum(comp_arr * min_value, arr)
        arr = np.minimum(comp_arr * max_value, arr)
        return arr

    @staticmethod
    def __gamma_correction(rgb, gamma=2.4):
        """
        Apply sRGB gamma correction
        :param rgb:
        :param gamma:
        :return: linear_rgb
        """
        linear_rgb = np.zeros_like(rgb, dtype=np.float16)
        for i in range(3):
            idx = rgb[:, :, i] > 0.04045 * 255
            linear_rgb[idx, i] = ((rgb[idx, i] / 255 + 0.055) / 1.055) ** gamma
            idx = np.logical_not(idx)
            linear_rgb[idx, i] = rgb[idx, i] / 255 / 12.92
        return linear_rgb

    @staticmethod
    def __inverse_gamma_correction(linear_rgb, gamma=2.4):
        """

        :param linear_rgb: array of shape (M, N, 3) with linear sRGB values between in the range [0, 1]
        :param gamma: float
        :return: array of shape (M, N, 3) with inverse gamma correction applied
        """
        rgb = np.zeros_like(linear_rgb, dtype=np.float16)
        for i in range(3):
            idx = linear_rgb[:, :, i] <= 0.0031308
            rgb[idx, i] = 255 * 12.92 * linear_rgb[idx, i]
            idx = np.logical_not(idx)
            rgb[idx, i] = 255 * (1.055 * linear_rgb[idx, i] ** (1 / gamma) - 0.055)
        return np.round(rgb)

    @staticmethod
    def deuteranopia_img(img) -> Image.Image:
        gamma = 0.1
        simulate_type = "d"

        img = np.asarray(img.convert("RGB"), dtype=np.float16)
        orig_img = Datonize.__gamma_correction(img, gamma)
        simul_rgb = Datonize.__simulate(orig_img, simulate_type)
        simul_img = Datonize.__array_to_img(simul_rgb, gamma)

        return simul_img


def cutout(*, mask_size=24, cutout_inside=False, mask_color=(255)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color

        return image

    return _cutout


def load_image(src_path, image_size=None):

    def _crop_background(numpy_src):

        def _get_vertex(img):
            index = 0
            for i, items in enumerate(img):
                if items.max() != 0:  # activate where background is '0'
                    index = i
                    break

            return index

        numpy_src_y1 = _get_vertex(numpy_src)
        numpy_src_y2 = len(numpy_src) - _get_vertex(np.flip(numpy_src, 0))
        numpy_src_x1 = _get_vertex(np.transpose(numpy_src))
        numpy_src_x2 = len(numpy_src[0]) - _get_vertex(np.flip(np.transpose(numpy_src), 0))

        return numpy_src_x1, numpy_src_y1, numpy_src_x2, numpy_src_y2

    pil_image = Image.open(src_path).convert('RGB')

    if image_size is not None:
        pil_image = pil_image.resize([image_size, image_size])

    pil_image = m_invert(pil_image)

    return pil_image


def load_image_deu(src_path):
    image = load_image(src_path)
    image_deu = Datonize.deuteranopia_img(image)

    return image, image_deu


def load_image_diff(src_path):
    image = load_image(src_path)
    image_deu = Datonize.deuteranopia_img(image)
    image_diff = make_diff(image, image_deu)

    return image, image_diff


def load_image_deu_stack(src_path):
    image = load_image(src_path)
    image_deu = Datonize.deuteranopia_img(image)
    image_diff = make_diff(image, image_deu)

    return image, image_deu, image_diff


def save_plt_figure(data, fn, label=None):
    plt.plot(data, label=label)
    if label is not None:
        plt.legend()
    plt.savefig('./outputs/' + fn)
    plt.clf()


def save_plt_figures(data, labels, fn):
    assert len(data) == len(labels), 'data and label length should be same'

    for i in range(len(data)):
        plt.plot(data[i], label=labels[i])

    plt.legend()
    plt.savefig('./outputs/' + fn)
    plt.xlabel('')
    plt.clf()


def m_rgb_to_hsv(src):
    if isinstance(src, Image.Image):
        r, g, b = src.split()

        h_dat = []
        s_dat = []
        v_dat = []

        for rd, gn, bl in zip(r.getdata(), g.getdata(), b.getdata()):
            h, s, v = colorsys.rgb_to_hsv(rd / 255., gn / 255., bl / 255.)
            h_dat.append(int(h * 255.))
            s_dat.append(int(s * 255.))
            v_dat.append(int(v * 255.))
        r.putdata(h_dat)
        g.putdata(s_dat)
        b.putdata(v_dat)

        return Image.merge('RGB', (r, g, b))
    else:
        return None


def m_hsv_to_rgb(src):
    if isinstance(src, Image.Image):
        r, g, b = src.split()

        h_dat = []
        s_dat = []
        v_dat = []

        for rd, gn, bl in zip(r.getdata(), g.getdata(), b.getdata()):
            h, s, v = colorsys.hsv_to_rgb(rd/255., gn/255., bl/255.)
            h_dat.append(int(h*255.))
            s_dat.append(int(s*255.))
            v_dat.append(int(v*255.))
        r.putdata(h_dat)
        g.putdata(s_dat)
        b.putdata(v_dat)

        return Image.merge('RGB', (r, g, b))
    else:
        return None


def numpy_to_pil(src):
    return Image.fromarray(np.uint8(src), 'RGB')


def tensor_to_pil(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0):

    ndarr = tensor_to_numpy(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                            normalize=normalize, range=range, scale_each=scale_each)
    pil_im = Image.fromarray(ndarr)

    return pil_im


def tensor_to_numpy(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0):

    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    return ndarr


def rgb_to_lab(src):
    """
    :param src: numpy array
    :return: numpy array
    """

    return color.rgb2lab(src)


def lab_to_rgb(src):
    """
    :param src: numpy array
    :return: numpy array
    """

    return color.lab2rgb(src)


def back_to_image(_image, file_name):
    to_numpy = lambda _image: _image.data.mul_(255).clamp_(0, 255).permute(1, 2, 0).cpu().numpy().astype('uint8')
    ndarr = to_numpy(_image)
    im = Image.fromarray(ndarr)
    im.save(file_name)


def apply_stencil_mask(_input, _output, _target) -> torch.Tensor:
    """
    :param _input: Tensor
    :param _output: Tensor
    :param _target: Tensor
    There is no bitwise function for FloatTensor in Pytorch.
    Compose stencil mask to make masking operation.
    """
    _input = _input.squeeze(0)[0:3, ]

    _input, _output, _target = _input.clone().cpu(), _output.clone().cpu(), _target.clone().cpu()
    stencil_mask = (_input != _target).float()
    masked_image = torch.where(stencil_mask == 1, _output, _input)

    return masked_image


def get_histogram(_output, _target, channel_num=3, bins=100, sigma=0.05):
    """
    :param _output: Tensor
    :param _target: Tensor
    :param channel_num: The number of channel
    :param bins: The number of 'x axis' in plot
    :param sigma: Gaussian Histogram indices
    """

    # https://www.nature.com/articles/ncomms13890
    # Quantum-chemical insights from deep tensor neural networks
    class GaussianHistogram(nn.Module):

        def __init__(self, bins, min, max, sigma):
            super(GaussianHistogram, self).__init__()
            self.bins = bins
            self.min = min
            self.max = max
            self.sigma = sigma
            self.delta = float(max - min) / float(bins)
            self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5).cuda()

        def forward(self, x):
            x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
            x = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
            x = x.sum(dim=1)
            return x

    eps = 1e-5
    _output = _output.squeeze(0).contiguous()
    _target = _target.squeeze(0).contiguous()

    output_hist_tensor = torch.zeros([channel_num, bins]).cuda()
    target_hist_tensor = torch.zeros([channel_num, bins]).cuda()

    for i in range(channel_num):

        gh_output = GaussianHistogram(bins=100, min=_output.min(), max=_output.max(), sigma=sigma)
        gh_target = GaussianHistogram(bins=100, min=_target.min(), max=_target.max(), sigma=sigma)

        output_hist_tensor[i] = gh_output(_output[i].view(-1))
        target_hist_tensor[i] = gh_target(_target[i].view(-1))

    _output_t = transforms.Normalize(mean=[_output[0].mean(), _output[1].mean(), _output[2].mean()],
                                     std=[_output[0].std() + eps, _output[1].std() + eps, _output[2].std() + eps])
    _target_t = transforms.Normalize(mean=[_target[0].mean(), _target[1].mean(), _target[2].mean()],
                                     std=[_target[0].std() + eps, _target[1].std() + eps, _target[2].std() + eps])

    _output = _output_t(_output)
    _target = _target_t(_target)

    return output_hist_tensor, target_hist_tensor


def save_histogram(_input, _output, _target, channel='abc', bins=100, data_name=None):
    """
    :param _input: Tensor
    :param _output: Tensor
    :param _target: Tensor
    :param channel: The name of channel
    :param bins: The number of 'x axis' in plot
    TODO: is this need _input Tensor?
    """

    # https://www.nature.com/articles/ncomms13890
    class GaussianHistogram(nn.Module):
        def __init__(self, bins, min, max, sigma):
            super(GaussianHistogram, self).__init__()
            self.bins = bins
            self.min = min
            self.max = max
            self.sigma = sigma
            self.delta = float(max - min) / float(bins)
            self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)

        def forward(self, x):
            x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
            x = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
            x = x.sum(dim=1)
            return x

    _input = _input.squeeze(0).contiguous().detach().cpu()
    _output = _output.squeeze(0).contiguous().detach().cpu()
    _target = _target.squeeze(0).contiguous().detach().cpu()

    input_hist = torch.zeros([len(channel), bins])
    output_hist = torch.zeros([len(channel), bins])
    target_hist = torch.zeros([len(channel), bins])

    for i in range(len(channel)):
        gh_input = GaussianHistogram(bins=bins, min=_input.min(), max=_input.max(), sigma=0.05)
        gh_output = GaussianHistogram(bins=bins, min=_output.min(), max=_output.max(), sigma=0.05)
        gh_target = GaussianHistogram(bins=bins, min=_target.min(), max=_target.max(), sigma=0.05)

        input_hist[i] = gh_input(_input[i].view(-1))
        output_hist[i] = gh_output(_output[i].view(-1))
        target_hist[i] = gh_target(_target[i].view(-1))

        data_list = [input_hist[i].numpy(), output_hist[i].numpy(), target_hist[i].numpy()]
        label_list = [channel[i] + '_input', channel[i] + '_output', channel[i] + '_target']

        if data_name is not None:
            save_plt_figures(data_list, labels=label_list, fn=data_name + '_histogram_' + channel + '_' + channel[i])
        else:
            save_plt_figures(data_list, labels=label_list, fn='histogram_' + channel + '_' + channel[i])

    # return output_hist_dim_tensor, target_hist_dim_tensor


def clip_by_threshold(_input, _output, threshold=0.1) -> torch.Tensor:
    """
    :param _input: Tensor
    :param _output: Tensor
    :param threshold: threshold of each value
    """

    _input_min = _input * (1 - threshold)
    _input_max = _input * (1 + threshold)

    _output = torch.max(torch.min(_output, _input_max), _input_min)

    return _output


def clip_tensor(_input, _output, _target) -> torch.Tensor:
    """
    :param _input: Tensor
    :param _output: Tensor
    :param _target: Tensor
    """

    assert _input.shape == _output.shape == _target.shape, 'Tensor shape must be same!'

    mask = _input > _target

    tensor_max = torch.max(_output, _target)
    tensor_min = torch.min(_output, _target)

    clamped_tensor = torch.where(mask, tensor_max, tensor_min)

    return clamped_tensor


def variational_prediction(_input, _output, _target, is_clip_tensor=True) -> torch.Tensor:
    """
    :param _input: Tensor
    :param _output: Tensor
    :param _target: Tensor
    :param is_clip_tensor: use 'clip_tensor'

    We would recommend to use this function with 'apply_stencil' True
    """

    output_v1 = _input * 2 - _output
    output_v2 = _output

    if is_clip_tensor:
        output_v1 = clip_tensor(_input, output_v1, _target)
        output_v2 = clip_tensor(_input, output_v2, _target)

    sim_v1 = F.mse_loss(output_v1, _target)
    sim_v2 = F.mse_loss(output_v2, _target)

    # choose the most relevant output to target
    _output = output_v1 if sim_v1 < sim_v2 else output_v2

    return _output.requires_grad_()


def feature_loss(feat_1, feat_2):

    def gradient_right(x):
        """
        :param x: shape[batch_size, 1d_tensor]
        """
        x = x.view([x.shape[0], -1])
        pad = nn.ConstantPad1d(1, 0)

        x_pad = pad(x)
        x_2 = x_pad[:, :-2]
        x_right = x - x_2

        return x_right[:, 1:]

    def gaussian(window_size, sigma):
        """
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        :param window_size: size of the SSIM sampling window e.g. 11
        :param sigma: Gaussian variance
        :returns: 1xWindow_size Tensor of Gaussian weights
        :rtype: Tensor
        """
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2 / 2

    cos_sim = F.cosine_similarity(feat_1.squeeze(0), feat_2.squeeze(0), dim=1)

    feat_1 = gradient_right(feat_1).unsqueeze(0)
    feat_2 = gradient_right(feat_2).unsqueeze(0)

    _1D_window = gaussian(feat_1.shape[2], 1.5).unsqueeze(0).unsqueeze(0).cuda()

    mu1 = F.conv1d(feat_1, _1D_window, padding=1 // 2, groups=1)
    mu2 = F.conv1d(feat_2, _1D_window, padding=1 // 2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv1d(feat_1 * feat_1, _1D_window, padding=1 // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv1d(feat_2 * feat_2, _1D_window, padding=1 // 2, groups=1) - mu2_sq
    sigma12 = F.conv1d(feat_1 * feat_2, _1D_window, padding=1 // 2, groups=1) - mu1_mu2

    sigma1_sq = max(0, sigma1_sq)
    sigma2_sq = max(0, sigma2_sq)

    sigma1 = math.sqrt(sigma1_sq)
    sigma2 = math.sqrt(sigma2_sq)

    l_error = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)     # Luminance error     same = 1, others to negative
    c_error = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)   # Contrast error  same = 1, others to negative
    s_error = (sigma12 + C3) / (sigma1 * sigma2 + C3)   # Structure error           same = 1, others to negative

    # l_error = (l_error + 1) / 2
    c_error = 1 - ((c_error + 1) / 2)
    s_error = 1 - ((s_error + 1) / 2)
    # cos_sim = (cos_sim + 1) / 2

    loss = c_error + s_error

    return loss


def make_diff(img, img_deu):
    """
    :param img: PIL Image
    :param img_deu: PIL Image with deuteranopia perspective
    """

    trans = torchvision.transforms.ToTensor()
    img1 = trans(img)
    img1_deu = trans(img_deu)

    img_diff = abs(img1 - img1_deu)
    img_diff = tensor_to_pil(img_diff)

    return img_diff


def m_invert(img):
    r"""Invert the input PIL Image.
    Args:
        img (PIL Image): Image to be inverted.
    Returns:
        PIL Image: Inverted image.
    """
    if not VF._is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if img.mode == 'RGBA':
        r, g, b, a = img.split()
        rgb = Image.merge('RGB', (r, g, b))
        inv = invert(rgb)
        r, g, b = inv.split()
        inv = Image.merge('RGBA', (r, g, b, a))
    elif img.mode == 'LA':
        l, a = img.split()
        l = invert(l)
        inv = Image.merge('LA', (l, a))
    else:
        inv = invert(img)

    return inv
