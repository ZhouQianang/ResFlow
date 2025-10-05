import numpy as np
import torch
import torch.utils.data as data

import random
import os
import glob

import imageio as iio


class DSECEVITestdataset(data.Dataset):
    def __init__(self, scene):
        super(DSECEVITestdataset, self).__init__()
        self.init_seed = False
        
        self.voxel_files = []
        self.events_files = []
        self.flow_files = []
        self.image1_files = []
        self.image2_files = []

        self.voxel_root = '/data/DSEC-Flow/DSEC_Event_v0_16bins/test/'
        self.images_root = '/data/DSEC-Flow/DSEC_EventImage_v1_5bins/test/'
        self.events_root = '/data/DSEC-Flow/DSEC_Events/test/'
        
        

        flow_ts = np.loadtxt(os.path.join(self.images_root,scene,'flow/forward_timestamps.txt'),delimiter=',', skiprows=1)
        images_ts = np.loadtxt(os.path.join(self.images_root,scene,'images/timestamps.txt'))
    
        for i, flowt in enumerate(flow_ts):

            idx = int(flow_ts[i][2])
        
            voxel_file = os.path.join(self.voxel_root,scene,f'{idx:06d}.npz')
            assert os.path.exists(voxel_file), f"The voxel file {voxel_file} not exist."

            events_file = os.path.join(self.events_root, scene, f'{idx:06d}.npz')
            assert os.path.exists(events_file), f"The events file {events_file} not exist."
        
            idx1 = np.where(images_ts == flow_ts[i][0])
            idx2 = np.where(images_ts == flow_ts[i][1])
            assert idx2[0].tolist()[0] - idx1[0].tolist()[0] == 2
        
            image1_file = os.path.join(self.images_root,scene,f'images/left/ev_inf/{idx1[0][0]:06d}.png')
            assert os.path.exists(image1_file), f"The file {image1_file} not exist."
            image2_file = os.path.join(self.images_root,scene,f'images/left/ev_inf/{idx2[0][0]:06d}.png')
            assert os.path.exists(image2_file), f"The file {image2_file} not exist."

            self.voxel_files.append(voxel_file)
            self.events_files.append(events_file)
            self.image1_files.append(image1_file)
            self.image2_files.append(image2_file)
        
        print('There has (', len(self.events_files),len(self.flow_files),len(self.image1_files),len(self.image2_files),1361*6,') samples in training')

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        voxelfile = self.voxel_files[index]
        eventfile = self.events_files[index]
        city = voxelfile.split('/')[-2]
        ind = voxelfile.split('/')[-1].split('.')[0].split('_')[-1]
        
        voxel_file = np.load(voxelfile)
        voxel1 = voxel_file['voxel_prev'][:, :, 1:]#.transpose([1,2,0])
        voxel2 = voxel_file['voxel_curr'][:, :, 1:]#.transpose([1,2,0])

        event_file = np.load(eventfile)
        events = event_file['events_prev']
        
        img1 = np.asarray(iio.imread(self.image1_files[index], format='PNG-FI'))
        img2 = np.asarray(iio.imread(self.image2_files[index], format='PNG-FI'))
        
        voxel1 = torch.from_numpy(voxel1).permute(2, 0, 1).float()
        voxel2 = torch.from_numpy(voxel2).permute(2, 0, 1).float()
        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        
        return voxel1, voxel2, img1, img2, events, city, ind
    
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




def make_DSEC_Eval_Test_datasets():

    test_scenes = ['zurich_city_12_a', 'zurich_city_14_c', 'zurich_city_15_a', 
                    'thun_01_b', 'thun_01_a', 'interlaken_01_a', 'interlaken_00_b']
    # test_scenes = ['zurich_city_14_c']
    
    val_datasets = []

    for scene in test_scenes:
        dset = DSECEVITestdataset(scene)
        val_datasets.append(dset)

    return val_datasets