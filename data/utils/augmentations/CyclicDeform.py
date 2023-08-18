import numpy as np
import colorsys
import torch
import math
from torchvision.transforms.functional import elastic_transform

class CyclicDeform:
    def __init__(self, epoch: int, cycle: int, img_size: tuple[int], device: torch.device, stretch_intensity=3):
        self.epoch = epoch
        assert isinstance(epoch, int)
        assert isinstance(cycle, int)
        assert isinstance(img_size, tuple)
        
        angle = (epoch*360/cycle)%360
        x = math.sin(math.radians(angle))
        y = math.cos(math.radians(angle))
        
        displacement_grid = np.mgrid[-1:1:img_size[0]*1j, -1:1:img_size[1]*1j]
        
        displacement_weights = np.abs(x*-displacement_grid[0]-(-displacement_grid[1])*(y))/np.linalg.norm((x,y))
        displacement_weights -= 1
        displacement_weights = np.abs(displacement_weights)
        displacement_weights /= displacement_weights.max()
        displacement_weights = np.around(displacement_weights, 2)
        # z_displacement = np.full((control_points[0],control_points[1]), 1)
        # self.displacement = torch.from_numpy(np.asarray((z_displacement, y*displacement_weights, x*displacement_weights))* stretch_intensity)
        self.displacement = torch.from_numpy(np.asarray((x*displacement_weights, y*displacement_weights))* stretch_intensity)
        self.displacement = torch.unsqueeze(self.displacement.permute((1,2,0)), 0).to(device)
        # self.displacement = torch.unsqueeze(self.displacement, 1)
        # print(self.displacement.shape)

    def __call__(self, sample: torch.tensor):
        # image = sample.permute(1, 2, 0).numpy()[:,:,0]
        # image = sample[0]
        # print(image.shape)
        # normalized = (image - torch.min(image))/np.ptp(image)
        # deformed = deform_grid(image, self.displacement)
        deformed = elastic_transform(sample, self.displacement)
        # deformed = elasticdeform.deform_grid(image.numpy(), self.displacement.numpy())
        return deformed
