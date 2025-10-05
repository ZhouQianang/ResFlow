import sys
sys.path.append('model')

import os
import imageio
from tqdm import tqdm
import numpy as np
import glob
import torch
import time

from utils.warp_event import WarpEventsEraftInterp
from datasets.DSEC_Eval_dataloader import make_DSEC_Eval_Test_datasets


@torch.no_grad()                   
def validate_event_fwl(model, val_num, vis_iwe=False, vis_flow=False, eval_dense=True):
    model.eval()
    val_datasets = make_DSEC_Eval_Test_datasets()
      
    # ewarp = WarpEventsEraftInterp(vis_IWE=vis_iwe, vis_Flow=vis_flow)
    ewarp = WarpEventsEraftInterp(vis_IWE=vis_iwe, vis_Flow=vis_flow)
    
    total_efwl=[]
    
    for val_dataset in val_datasets:
        
        
        efwl_list = []
        
        bar = tqdm(enumerate(val_dataset),total=len(val_dataset), ncols=60)

        for index, (voxel1, voxel2, img1, img2,events, voxel2dp,  city, ind) in bar:
        # for index, (voxel1, voxel2, voxel2dp, img1, img2, city, ind) in bar:
            
            if index > val_num:
                break
            
            voxel1 = voxel1[None].cuda()
            voxel2 = voxel2[None].cuda()
            voxel2dp = voxel2dp[None].cuda()
            img1 = img1[None].cuda()
            img2 = img2[None].cuda()
            

            flow_pred = model(voxel1, voxel2)  # [B*5,2,H,W]
            
            flow_dense = list(flow_pred['dense_pred'].chunk(5, dim=0))
            flow_global = flow_pred['global_pred']
            flow_linear = [flow_global*0.2, flow_global*0.4, flow_global*0.6, flow_global*0.8, flow_global]
            
            if eval_dense:
                fwl = ewarp.calculate_fwl_loss(events, flow_dense, f"{city}_{ind}")
            else:
                fwl = ewarp.calculate_fwl_loss(events, flow_linear, f"{city}_{ind}")
            
            efwl_list.append(np.array(fwl.cpu().numpy()))
        
        efwl = np.mean(np.array(efwl_list), axis=0)
    
        print("\n", city, ", event-fwl, ", efwl, "\n")
        
        total_efwl +=  efwl_list
    
    efwl_mean = np.mean(np.array(total_efwl), axis=0)
    
    print("\n Overall  event-fwl,  ", efwl_mean, "sample numbers: ", len(total_efwl))
    
    return {
        'efwl':efwl_mean,
    }

@torch.no_grad()
def evaluate_TDF_DSEC(model, test_save):
    model.eval()
    val_datasets = make_DSEC_Eval_Test_datasets()
    
    time_list = []
    
    for val_dataset in val_datasets[0:1]:
        bar = tqdm(enumerate(val_dataset),total=len(val_dataset), ncols=60)
        
        for index, (voxel1, voxel2, img1, img2, events, voxel2dp, city, ind) in bar:
            
            voxel1 = voxel1[None].cuda()
            voxel2 = voxel2[None].cuda()
            img1 = img1[None].cuda()
            img2 = img2[None].cuda()
            
            # print(voxel1.shape,voxel1.max(), voxel2.max(), voxel1.min(),voxel2.min(),img1.max(),img1.min())
        
            start = time.time()
            flow_pred = model(voxel1, voxel2)#[1*5, 2, H, W]
            # flow_last = flow_pred['dense_pred'].chunk(5, dim=0)[-1]
            # flow_last = flow_pred['global_repred']
            # flow_last = flow_pred['global_pred']
            # flow_last = flow_pred
            
            end = time.time()
            time_list.append((end-start)*1000)
            
            continue
    
    avg_time = sum(time_list)/len(time_list)
    print(f'Time: {avg_time} ms.')  
    print('Done!')

    
    
    
from utils.flow_viz import flow_to_image
@torch.no_grad()
def vis_noise_resflow(model, vis_dir, noise_type='region'):
    
    os.makedirs(vis_dir, exist_ok=True)
    
    model.eval()
    val_datasets = make_DSEC_Eval_Test_datasets()
    
    
    for val_dataset in val_datasets[1:2]:
        
        bar = tqdm(enumerate(val_dataset),total=len(val_dataset), ncols=60)

        for index, (voxel1, voxel2, img1, img2,events, voxel2dp,  city, ind) in bar:
            
            # if index > 12:
            #     break
            
            voxel1 = voxel1[None].cuda()
            voxel2 = voxel2[None].cuda()

            flow_pred = model(voxel1, voxel2, add_noise=True, noise_type=noise_type)  # [B*5,2,H,W]
            
            flow_init = flow_pred['global_pred']
            noise = flow_pred['noise']
            resflows = flow_pred['resflows']
            flow_refine = flow_pred['dense_pred'].chunk(5, dim=0)[-1]
            
            noise_img = flow_to_image(noise.squeeze().permute(1,2,0).cpu().numpy())
            noise_pth = os.path.join(vis_dir, f'{city}_{ind}_2_noise.png')
            imageio.imwrite(noise_pth, noise_img)
            
            resf_img = flow_to_image(resflows.squeeze().permute(1,2,0).cpu().numpy())
            resf_pth = os.path.join(vis_dir, f'{city}_{ind}_3_resflow.png')
            imageio.imwrite(resf_pth, resf_img)
            np.save('resflow', resflows.squeeze().permute(1,2,0).cpu().numpy())
            
            init_img = flow_to_image(flow_init.squeeze().permute(1,2,0).cpu().numpy())
            init_pth = os.path.join(vis_dir, f'{city}_{ind}_1_init.png')
            imageio.imwrite(init_pth, init_img)
            
            final_img = flow_to_image(flow_refine.squeeze().permute(1,2,0).cpu().numpy())
            final_pth = os.path.join(vis_dir, f'{city}_{ind}_4_final.png')
            imageio.imwrite(final_pth, final_img)