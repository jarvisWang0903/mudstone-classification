import math
import numbers
import random
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img):

        for a in self.augmentations:
            img = a(img)
        return np.array(img)


class RandomCrop(object):

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        tw, th = self.size
        if w == tw and h == th:
            return img

        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):

        w, h = img.size
        tw, th = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img):
        return img.resize(self.size, Image.BILINEAR)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):

        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR)


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):

        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))

                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR)
        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR)


class RandomSized_and_Crop(object):
    def __init__(self, size):
        self.size = size
        # self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img):

        h = int(random.uniform(0.7, 1) * img.size[0])
        w = int(random.uniform(0.7, 1) * img.size[1])


        img = img.resize((w, h), Image.BILINEAR)

        return self.crop(img)


class RandomSized_and_CenterCrop(object):
    def __init__(self, size):
        self.size = size
        # self.scale = Scale(self.size)
        self.crop = CenterCrop(self.size)

    def __call__(self, img):


        w = int(random.uniform(0.7, 1) * img.size[0])
        h = int(random.uniform(0.7, 1) * img.size[1])

        img = img.resize((w, h), Image.BILINEAR)

        return self.crop(img)