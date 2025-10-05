import sys
sys.path.append('model')
import time
import os
import random
from tqdm import tqdm
import wandb
import torch
import numpy as np
from utils.file_utils import get_logger
from datasets.DSECdataloader import make_DsecS_train_loader

####Important####
from model.TDF_noscale import TDF_noscale
from model.TDF import TDF
####Important####

from evaluate_dsec import validate_event_fwl

from tensorboardX import SummaryWriter
from utils.logger import Logger

MAX_FLOW = 400
SUM_FREQ = 100

class Loss_Tracker:
    def __init__(self, wandb):
        self.running_loss = {}
        self.total_steps = 0
        self.wandb = wandb
    def push(self, metrics):
        self.total_steps += 1
        
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            
            self.running_loss[key] += metrics[key]
        
        if self.total_steps % SUM_FREQ == 0:
            if self.wandb:
                wandb.log({'EPE': self.running_loss['epe']/SUM_FREQ}, step=self.total_steps)
            self.running_loss = {}
            

def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        print(classname," has been frozen")
        m.eval()
            

class Trainer:
    def __init__(self, args):
        self.args = args

        if self.args.noscale:
            self.model = TDF_noscale(input_bins=15, 
                            noise_weight=args.noise_weight,
                            param_frozen=args.global_frozen,
                            ckpt_path=args.ckpt_path)
            print("Without scaled.")
        else:
            self.model = TDF(input_bins=15, 
                            noise_weight=args.noise_weight,
                            param_frozen=args.global_frozen,
                            ckpt_path=args.ckpt_path)
            print("Scaled.")

        self.model = self.model.cuda()
        print("resume train from: ",args.ckpt_path)

        #Loader
        self.train_loader = make_DsecS_train_loader(args.batch_size, args.num_workers)
        print('train_loader done!')

        #Optimizer and scheduler for training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=0.0001
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=args.lr,
            total_steps=args.num_steps + 100,
            pct_start=0.5,
            cycle_momentum=False,
            anneal_strategy='linear')
        
        self.checkpoint_dir = args.checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.writer = get_logger(os.path.join(self.checkpoint_dir, 'train.log'))
        self.tracker = Loss_Tracker(args.wandb)

        self.writer.info('====A NEW TRAINING PROCESS====')
        
        self.summary_writer = SummaryWriter("/output/logs/")
        
        self.val_num = 300

    def train(self):
        # self.writer.info(self.model)
        self.writer.info(self.args)
        self.model.train()
        
        if self.args.bn_frozen:
            self.model.apply(freeze_bn)
            
        self.logger = Logger(self.scheduler, self.summary_writer,SUM_FREQ,start_step=0)
        
        total_steps = 0
        keep_training = True
        add_noise = self.args.add_noise

        if not self.args.selfgt:
            print("use real gt for supv.")
        else:
            print("self-supv.")
        
        print("Training Details: ", "noise_weight:",self.args.noise_weight, "param_frozen:",self.args.global_frozen,"bn_frozen:",self.args.bn_frozen,"add_noise:",self.args.add_noise, "loss_weight:", self.args.loss_weight,"------------------------------------------------------------")
        VALFREQ=2000
        while keep_training:

            bar = tqdm(enumerate(self.train_loader),total=len(self.train_loader), ncols=60)
            for index, (voxel1, voxel2, flowmap, valid) in bar:
                self.optimizer.zero_grad()
                flow_preds = self.model(voxel1.cuda(), voxel2.cuda(), add_noise=add_noise, noise_type=self.args.noise_type) # (N), [B*5, 2, H, W]
                if not self.args.selfgt:
                    flow_loss, loss_metrics = sequence_loss(flow_preds['dense_pred_list'], flowmap.cuda(), valid.cuda(), self.args.weight, MAX_FLOW)
                else:
                    flow_loss, loss_metrics = sequence_loss_self(flow_preds['dense_pred_list'], flow_preds['global_pred'], self.args.weight, MAX_FLOW)
                
                ## global
                flow_glb_loss, loss_metrics = sequence_loss_LTR(flow_preds['global_pred_list'], flowmap.cuda(), valid.cuda(), self.args.weight, MAX_FLOW)
                
                flow_loss = flow_loss + flow_glb_loss

                flow_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                
                # summary_write
                self.logger.push(loss_metrics)

                bar.set_description(f'Step: {total_steps}/{self.args.num_steps}')
                self.tracker.push(loss_metrics)
                total_steps += 1


                results = {}
                
                if total_steps % VALFREQ == 0:
                    results.update(validate_event_fwl(self.model, self.val_num))
                    self.logger.push(results)
                    self.model.train()

                if (total_steps % VALFREQ == 0):
                    ckpt = os.path.join(self.args.checkpoint_dir, f'checkpoint_{total_steps}.pth')
                    torch.save(self.model.state_dict(), ckpt)
                if total_steps > self.args.num_steps:
                    keep_training = False
                    break
            
            time.sleep(0.03)
        ckpt_path = os.path.join(self.args.checkpoint_dir, 'checkpoint.pth')
        torch.save(self.model.state_dict(), ckpt_path)
        return ckpt_path
      

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions 
    
    Args:
        - flow_preds ... (N), [B*5, 2, H, W]
        - flow_gt ... [B, 2, H, W]
    Returns:
        - flow_loss ... [B]
        - metrics ... [B, 4]
    """

    n_predictions = len(flow_preds)

    flow_loss = 0.0

    mag = torch.sum(flow_gt**2, dim=1).sqrt()#b,h,w
    valid = (valid>0.5) & (mag < max_flow)#b,1,h,w

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        flow = flow_preds[i].chunk(5,dim=0)[-1]
        i_loss = (flow - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def sequence_loss_LTR(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions 
    
    Args:
        - flow_preds ... (N), [B, 2, H, W]
        - flow_gt ... [B, 2, H, W]
    Returns:
        - flow_loss ... [B]
        - metrics ... [B, 4]
    """

    n_predictions = len(flow_preds)

    flow_loss = 0.0


    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()#b,h,w
    valid = (valid>0.5) & (mag < max_flow)#b,1,h,w

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        flow = flow_preds[i]
        # print(flow.shape)
        i_loss = (flow - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def sequence_loss_self(flow_preds, flow_gt, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions 
    
    Args:
        - flow_preds ... (N), [B*5, 2, H, W]
        - flow_gt ... [B, 2, H, W]
    Returns:
        - flow_loss ... [B]
        - metrics ... [B, 4]
    """

    n_predictions = len(flow_preds)

    flow_loss = 0.0


    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()#b,h,w
    valid = (mag < max_flow)#b,1,h,w

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        flow = flow_preds[i].chunk(5,dim=0)[-1]
        i_loss = (flow - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()
        
    # print("flow_gt shape", flow_gt.shape, valid.shape, flow.shape)

    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


        
if __name__=='__main__':
    import argparse


    parser = argparse.ArgumentParser(description='TDF')
    #training setting
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--checkpoint_dir', type=str, default='')
    parser.add_argument('--lr', type=float, default=2e-4)

    #datasets setting
    parser.add_argument('--crop_size', type=list, default=[288, 384])

    #dataloader setting
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=8)

    #model setting
    parser.add_argument('--grad_clip', type=float, default=1)

    # loss setting
    parser.add_argument('--weight', type=float, default=0.8)

    #wandb setting
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--ckpt_path', type=str, default=None)
    
    ## new params
    parser.add_argument('--global_frozen', action='store_true', default=False)
    parser.add_argument('--bn_frozen', action='store_true', default=False)
    parser.add_argument('--add_noise', action='store_true', default=False)
    parser.add_argument('--noscale', action='store_true', default=False)
    parser.add_argument('--selfgt', action='store_true', default=False)
    parser.add_argument('--noise_weight', type=float, default=0.1)
    parser.add_argument('--noise_type', type=str, default='region')
    parser.add_argument('--loss_weight', type=float, default=0.5)
    
    args = parser.parse_args()
    set_seed(1)

    trainer = Trainer(args)
    trainer.train()
    
