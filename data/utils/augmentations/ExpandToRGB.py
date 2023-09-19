import numpy as np
import colorsys
import torch
import math
from torchvision.transforms.functional import elastic_transform

class ExpandToRGB:
    def __init__(self):
        a = 1
        # print(self.displacement.shape)

    def __call__(self, sample: torch.tensor):
        ret = sample.repeat_interleave(3, dim=1)
        # deformed = elasticdeform.deform_grid(image.numpy(), self.displacement.numpy())
        return ret
