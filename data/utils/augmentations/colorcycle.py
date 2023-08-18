import colorsys
import math
import torch

class CycleColor:
    def __init__(self, epoch: int, cycle: int, tolerance: int, device: torch.device):
        self.epoch = epoch
        self.background_hue = ((self.epoch+10)*360/cycle)%360
        self.digit_hue = ((self.epoch+30)*360/cycle)%360
        self.tolerance = tolerance
        self.rgb_background = torch.tensor(colorsys.hsv_to_rgb(math.radians(self.background_hue), 0.5, 0.5)).to(device)
        self.rgb_digit = torch.tensor(colorsys.hsv_to_rgb(math.radians(self.digit_hue), 0.5, 0.5)).to(device)
        # print(self.rgb_digit)
    def __call__(self, sample: torch.tensor):
        max = torch.max(sample)
        min = torch.min(sample)
        image = (sample-min)/(max-min)

        background_mask = (image < image.min()+self.tolerance) & (image > image.min()-self.tolerance)
        digit_mask = ~background_mask
        color_pic = torch.concat((image,)*3, axis=0)
        color_pic[:,background_mask.squeeze()]  = self.rgb_background.unsqueeze(1)
        color_pic[:,digit_mask.squeeze()]  = self.rgb_digit.unsqueeze(1)
        return color_pic
