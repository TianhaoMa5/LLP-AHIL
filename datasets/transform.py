import numpy as np
import torch
import cv2

import random

class PadandRandomCrop(object):
    '''
    Input tensor is expected to have shape of (H, W) for grayscale images.
    '''
    def __init__(self, border=4, cropsize=(32, 32)):
        self.border = border
        self.cropsize = cropsize

    def __call__(self, im):
        if len(im.shape) == 3:  # 彩色图像
            borders = [(self.border, self.border), (self.border, self.border), (0, 0)]
        elif len(im.shape) == 2:  # 灰度图像
            borders = [(self.border, self.border), (self.border, self.border)]

        convas = np.pad(im, borders, mode='reflect')
        H, W = convas.shape[:2]
        h, w = self.cropsize
        dh, dw = max(0, H-h), max(0, W-w)
        sh, sw = np.random.randint(0, dh), np.random.randint(0, dw)
        out = convas[sh:sh+h, sw:sw+w, ...]  # 适用于彩色和灰度图像
        return out



class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im):
        if np.random.rand() < self.p:
            if im.ndim == 3:
                # 三维数组（彩色图像），水平翻转
                im = im[:, ::-1, :]
            elif im.ndim == 2:
                # 二维数组（灰度图像），水平翻转
                im = im[:, ::-1]
        return im


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, im):
        im = cv2.resize(im, self.size)
        return im


class Normalize(object):
    '''
    Inputs are pixel values in range of [0, 255]. For grayscale images, mean and std are single values.
    '''

    def __init__(self, mean, std):
        # 对于灰度图像，mean和std是单值
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 1)

    def __call__(self, im):
        if im.ndim == 4:
            # 批量处理的图像，保持不变
            pass
        elif im.ndim == 3:
            # 单张彩色图像，检查是否真的有3个通道
            assert im.shape[2] == 3, "Input image has more than one channel but not 3 channels."
        elif im.ndim == 2:
            # 单张灰度图像，添加一个通道维度
            im = im[None, ...]
        else:
            raise ValueError(f"Unsupported image dimension: {im.ndim}")

        # 归一化处理
        im = im.astype(np.float32) / 255.0
        im -= self.mean
        im /= self.std
        return im


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, im):
        if len(im.shape) == 4:
            return torch.from_numpy(im.transpose(0, 3, 1, 2))
        elif len(im.shape) == 3:
            return torch.from_numpy(im.transpose(2, 0, 1))
        elif len(im.shape) == 2:
            # 处理2维图像（例如：灰度图像，如MNIST）
            return torch.from_numpy(im[None, :, :])

class Compose(object):
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, im):
        for op in self.ops:
            im = op(im)
        return im
class RandomAffine(object):
    '''
    Apply random affine transformation to the image.
    '''
    def __init__(self, degrees=15, translate=(0.1, 0.1), scale_range=(0.9, 1.1)):
        self.degrees = degrees
        self.translate = translate
        self.scale_range = scale_range

    def __call__(self, img):
        # Image dimensions
        h, w = img.shape[:2]

        # Rotation
        angle = random.uniform(-self.degrees, self.degrees)

        # Scale
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        # Translation (shift)
        max_dx = self.translate[0] * w
        max_dy = self.translate[1] * h
        dx = random.uniform(-max_dx, max_dx)
        dy = random.uniform(-max_dy, max_dy)

        # Transformation matrix
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += dx
        M[1, 2] += dy

        # Apply affine transformation
        transformed_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        return transformed_img