import colorsys
import math
import torch

class CycleColor:
    def __init__(self, epoch: int, cycle: int, background_tolerance: int, device: torch.device, generation_range: int, style_count: int):
        self.epoch = epoch
        self.background_hue = ((self.epoch+10)*360/cycle)%360
        self.digit_hue = ((self.epoch+30)*360/cycle)%360

        bg_saturation, gv_val, digit_saturation, digit_val = torch.FloatTensor(4, style_count).uniform_(0.2, 1)
        

        self.background_pool = torch.as_tensor([colorsys.hsv_to_rgb(math.radians(i), bg_saturation[index].item(), gv_val[index].item()) for index, i in enumerate(torch.linspace(self.background_hue, self.background_hue+generation_range, style_count))]).to(device)
        self.digit_pool = torch.as_tensor([colorsys.hsv_to_rgb(math.radians(i), digit_saturation[index].item(), digit_val[index].item()) for index, i in enumerate(torch.linspace(self.digit_hue+30, self.digit_hue+generation_range+30, style_count))]).to(device)
        self.style_count = style_count
        self.background_tolerance = background_tolerance
        # self.rgb_background = torch.tensor(colorsys.hsv_to_rgb(math.radians(self.background_hue), 0.5, 0.5)).to(device)
        # self.rgb_digit = torch.tensor(colorsys.hsv_to_rgb(math.radians(self.digit_hue), 0.5, 0.5)).to(device)
        # print(self.rgb_digit)
    def __call__(self, sample: torch.tensor):
        max = torch.max(sample)
        min = torch.min(sample)
        image = (sample-min)/(max-min)

        

        background_mask = (image < image.min()+self.background_tolerance) & (image > image.min()-self.background_tolerance)
        digit_mask = ~background_mask
        color_pic = image.repeat_interleave(3, dim=1)
        permuted_color_pic = color_pic.permute((0,2,3,1))
        
        bg_interleave_factors = background_mask.squeeze().sum(dim=(1,2))
        digit_interleave_factors = digit_mask.squeeze().sum(dim=(1,2))

        bg_color_indices = torch.randint(0,self.style_count, (sample.shape[0],))
        digit_color_indices = torch.randint(0,self.style_count, (sample.shape[0],))
        # print(self.background_pool[color_indices].shape, bg_interleave_factors.shape)
        bg_colors = self.background_pool[bg_color_indices].repeat_interleave(bg_interleave_factors, dim=0)
        digit_colors = self.digit_pool[digit_color_indices].repeat_interleave(digit_interleave_factors, dim=0)

        # print(bg_colors.shape, permuted_color_pic[background_mask.squeeze()].shape)

        permuted_color_pic[background_mask.squeeze()]  = bg_colors
        permuted_color_pic[digit_mask.squeeze()]  = digit_colors
        

        return color_pic
