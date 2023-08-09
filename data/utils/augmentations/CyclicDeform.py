import numpy as np
import colorsys
import torch
import math
import elasticdeform

class CyclicDeform:
    def __init__(self, epoch: int, cycle: int, control_points: tuple[int], stretch_intensity=3):
        self.epoch = epoch
        assert isinstance(epoch, int)
        assert isinstance(cycle, int)
        assert isinstance(control_points, tuple)
        
        angle = (epoch*360/cycle)%360
        x = math.sin(math.radians(angle))
        y = math.cos(math.radians(angle))
        
        displacement_grid = np.mgrid[-1:1:control_points[0]*1j, -1:1:control_points[1]*1j]
        
        displacement_weights = np.abs(x*-displacement_grid[0]-(-displacement_grid[1])*(y))/np.linalg.norm((x,y))
        displacement_weights -= 1
        displacement_weights = np.abs(displacement_weights)
        displacement_weights /= displacement_weights.max()
        displacement_weights = np.around(displacement_weights, 2)

        self.displacement = np.asarray((y*displacement_weights, x*displacement_weights))* stretch_intensity

    def __call__(self, sample: torch.tensor):
        image = sample.permute(1, 2, 0).numpy()
        normalized = (image - np.min(image))/np.ptp(image)

        deformed = elasticdeform.deform_grid(normalized, self.displacement)
        return deformed
