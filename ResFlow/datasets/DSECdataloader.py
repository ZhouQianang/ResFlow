import numpy as np
import torch
import torch.utils.data as data

import random
import os
import glob

import imageio as iio

from .Eaugument import Augumentor

class DSECdataset(data.Dataset):
    def __init__(self, augument=True):
        super(DSECdataset, self).__init__()
        self.init_seed = False
        
        self.events_files = []
        self.flow_files = []

        self.events_root = '/data/DSEC-Flow/DSEC_Event_v0_16bins/train'
        self.images_root = '/data/DSEC-Flow/DSEC_EventImage_v1_5bins/train/'
        
        self.augment = augument
        if self.augment:
            self.augmentor = Augumentor(crop_size=[288, 384])
        
        # load all scenes
        scenes = [d for d in os.listdir(self.events_root) if os.path.isdir(os.path.join(self.events_root, d))]

        for scene in scenes:
            flow_ts = np.loadtxt(os.path.join(self.images_root,scene,'flow/forward_timestamps.txt'),delimiter=',', skiprows=1)
    
            for i, flowt in enumerate(flow_ts):
        
                events_file = os.path.join(self.events_root,scene,f'{i:06d}.npz')
                assert os.path.exists(events_file), f"The file {events_file} not exist."
        
                flow_file = os.path.join(self.events_root,scene,f'flow_{i:06d}.npy')
                assert os.path.exists(flow_file), f"The file {flow_file} not exist."
        
                self.events_files.append(events_file)
                self.flow_files.append(flow_file)
        
        print('There has (', len(self.events_files),len(self.flow_files),1361*6,') samples in training')

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        
        voxel_file = np.load(self.events_files[index])
        voxel1 = voxel_file['voxel_prev'][:, :, 1:]#.transpose([1,2,0])
        voxel2 = voxel_file['voxel_curr'][:, :, 1:]#.transpose([1,2,0])
        
        # print("Data shape:",voxel1.shape,voxel2.shape,img1.shape,img2.shape)

        flow_16bit = np.load(self.flow_files[index])
        flow_map, valid2D = flow_16bit_to_float(flow_16bit)

        voxel1, voxel2, flow_map, valid2D = self.augmentor(voxel1, voxel2, flow_map, valid2D)
        
        voxel1 = torch.from_numpy(voxel1).permute(2, 0, 1).float()
        voxel2 = torch.from_numpy(voxel2).permute(2, 0, 1).float()
        
        
        flow_map = torch.from_numpy(flow_map).permute(2, 0, 1).float()
        valid2D = torch.from_numpy(valid2D).float()
        return voxel1, voxel2, flow_map, valid2D
    
    def __len__(self):
        return len(self.events_files)
    
def flow_16bit_to_float(flow_16bit: np.ndarray):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)
    valid_map = np.where(valid2D)

    # to actually compute something useful:
    flow_16bit = flow_16bit.astype('float')

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128
    return flow_map, valid2D


def make_DsecS_train_loader(batch_size, num_workers):
    dset = DSECdataset()
    loader = data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True)
    return loader

