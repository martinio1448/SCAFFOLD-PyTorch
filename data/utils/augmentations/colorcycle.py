import numpy as np
import colorsys
import torch
import math

class CycleColor:
    def __init__(self, epoch: int, cycle: int, tolerance: int):
        self.epoch = epoch
        self.background_hue = ((self.epoch+10)*360/cycle)%360
        self.digit_hue = ((self.epoch+30)*360/cycle)%360
        self.tolerance = tolerance

    def __call__(self, sample: np.ndarray):
        print("color")
        image = sample

        background_mask = (image < image.min()+self.tolerance) & (image > image.min()-self.tolerance)
        digit_mask = ~background_mask
        normalized = (image - np.min(image))/np.ptp(image)
        color_pic = np.concatenate((normalized,)*3, axis=2)

        color_pic[background_mask.squeeze(), :]  = colorsys.hsv_to_rgb(math.radians(self.background_hue), 0.5, 0.5)
        color_pic[digit_mask.squeeze(), :]  = colorsys.hsv_to_rgb(math.radians(self.digit_hue), 0.5, 0.5)
        return color_pic
