import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps
from torchvision.transforms import InterpolationMode
import random



class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)

class GBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ContrastiveLearningViewGenerator(object):
    def __init__(self, num_patch = 4):

        self.num_patch = num_patch
        
    def __call__(self, x):
        aug_transform =  transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=(0.25, 0.25), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GBlur(p=0.1),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            
        ])

        
        augmented_x = [aug_transform(x) for i in range(self.num_patch)]
       
        
        return augmented_x
    

class SimSiamViewGenerator(object):
    def __init__(self):
        pass
          
    def __call__(self, x):

        normalize = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([SimSiamGaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        augmented_x = [aug_transform(x) for i in range(2)]
        return augmented_x

class SimSiamGaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
 
        
