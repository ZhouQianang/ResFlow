import torch
import torch.nn.functional as F
import numpy as np


def bilinear_sampler(img, coords, mode='bilinear', padding_mode='zeros', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """

    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(wd, device=device), torch.arange(ht, device=device), indexing='xy')
    coords = torch.stack(coords, dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
    
class SmoothNoise:
    def __init__(self, scale=6, mode='bilinear'):
        
        self.scale = scale
        self.mode = mode
        self.add_prob = 0.6
        self.noise_min = -2
        self.noise_max = 2
        
        
    def add_noise(self, flow, weight=0.3):
        
        if np.random.rand() > self.add_prob:
            return flow, torch.zeros_like(flow)
            
        B, C, H, W = flow.shape
        
        noise = torch.randn([B, C, H//self.scale, W//self.scale]).to(flow.device)
        noise = torch.clamp(noise, min=self.noise_min, max=self.noise_max)
        
        # print(noise.shape, flow.shape, "******")
        upsampled_noise = F.interpolate(noise, size=[flow.shape[2], flow.shape[3]], mode=self.mode, align_corners=True)
        
        # print("add region noise")
        
        return flow + weight * upsampled_noise, weight*upsampled_noise
