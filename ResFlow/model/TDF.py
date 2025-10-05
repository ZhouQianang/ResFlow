import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import ExtractorF, ExtractorC
from corr import CorrBlock
from aggregate import MotionFeatureEncoder, MPA
from update import UpdateBlock, DenseUpdateBlockSeparate
from util import coords_grid, SmoothNoise

add_region_noise = SmoothNoise()

class TDF(nn.Module):
    def __init__(self, 
                input_bins=15, 
                noise_weight=0.3,
                param_frozen=False,
                ckpt_path=None):
        super(TDF, self).__init__()

        f_channel = 128
        self.split = 5
        self.corr_level = 1
        self.corr_radius = 3
        self.param_frozen = param_frozen
        self.noise_weight = noise_weight

        self.fnet = ExtractorF(input_channel=input_bins//self.split, outchannel=f_channel, norm='IN')
        self.cnet = ExtractorC(input_channel=input_bins//self.split + input_bins, outchannel=256, norm='BN')

        self.mfe = MotionFeatureEncoder(corr_level=self.corr_level, corr_radius=self.corr_radius)
        self.mpa = MPA(d_model=128)

        self.update = UpdateBlock(hidden_dim=128, split=(self.split))

        # for dense part
        self.dSupdate = DenseUpdateBlockSeparate(hidden_dim=128, split=self.split)
        
        # load pretrain part
        print("-----------------pretrain details-----------------------")
        if ckpt_path is None:
            print("Init from zeros!")
            print("----------------------------------------------------------")
            return
        else:
            print("Load pretrain model from: ",ckpt_path) 
            pretrained_model = torch.load(ckpt_path)

        # fnet_ev
        self.fnet.load_state_dict({k[5:]: v for k, v in pretrained_model.items() if k.startswith('fnet.')})
        # cnet_ev
        self.cnet.load_state_dict({k[5:]: v for k, v in pretrained_model.items() if k.startswith('cnet.')})
        # mfe
        self.mfe.load_state_dict({k[4:]: v for k, v in pretrained_model.items() if k.startswith('mfe.')})
        # mpa
        self.mpa.load_state_dict({k[4:]: v for k, v in pretrained_model.items() if k.startswith('mpa.')})
        # update
        self.update.load_state_dict({k[7:]: v for k, v in pretrained_model.items() if k.startswith('update.')})

        print("Global part has been loaded!")
        
        
        ### frozen params (backbone) or don't forzen all
        if self.param_frozen:
            for param in self.fnet.parameters():
                param.requires_grad = False
            for param in self.cnet.parameters():
                param.requires_grad = False
            for param in self.mfe.parameters():
                param.requires_grad = False
            for param in self.mpa.parameters():
                param.requires_grad = False
            for param in self.update.parameters():
                param.requires_grad = False
            print("Global param has been forzen!")
        else:
            print("Do not use forzen!")
        

        ######### Done!
        print("----------------------------------------------------------")

    def upsample_flow(self, flow, mask, scale=8):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(scale * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, scale*H, scale*W)
    
    def linear_upsample_flow(self, flow, upsample_factor=8):
        up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor

        return up_flow


    def forward(self, x1, x2, iters=6, diters=2, add_noise=False, noise_type='region'):

        flow_result = {}

        b,_,h,w = x2.shape

        #Feature maps [f_0 :: f_i :: f_g]
        voxels2 = x2.chunk(self.split, dim=1)
        voxelref = x1.chunk(self.split, dim=1)[-1]
        voxels = (voxelref,) + voxels2 #[group+1] elements
        fmaps = self.fnet(voxels)#Tuple(f0, f1, ..., f_g)
        

        # Context map [net, inp]
        # print(m1.shape,len(voxels2),voxels2[0].shape,m2.shape)
        cmap = self.cnet(torch.cat(voxels, dim=1))
        net, inp = torch.split(cmap, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)


        coords0 = coords_grid(b, h//8, w//8, device=cmap.device)
        coords1 = coords_grid(b, h//8, w//8, device=cmap.device)

        #MidCorr
        corr_fn_list = []
        for i in range(self.split):
            corr_fn = CorrBlock(fmaps[0], fmaps[i+1], num_levels=self.corr_level, radius=self.corr_radius) #[c01,c02,...,c05]
            corr_fn_list.append(corr_fn)

        flow_predictions = []
        for iter in range(iters):

            coords1 = coords1.detach()
            flow = coords1 - coords0

            corr_map_list = []
            du = flow/self.split 
            for i in range(self.split):
                coords = (coords0 + du*(i+1)).detach()
                corr_map = corr_fn_list[i](coords)
                corr_map_list.append(corr_map)

            corr_maps = torch.cat(corr_map_list, dim=0) 

            mfs = self.mfe(torch.cat([flow]*(self.split), dim=0), corr_maps)
            mfs = mfs.chunk((self.split), dim=0)
            mfs = self.mpa(mfs)
            mf = torch.cat(mfs, dim=1)
            net, dflow, upmask = self.update(net, inp, mf)
            coords1 = coords1 + dflow
            
            if self.training:
                flow_up = self.upsample_flow(coords1 - coords0, upmask)
                flow_predictions.append(flow_up)

        if self.training:
            flow_result.update({'global_pred_list': flow_predictions})
        # if not self.training:
        flow_result.update({'global_pred': self.upsample_flow(coords1 - coords0, upmask)})
        # return flow_result
        ################### for the dense predict #################
        coords1 = coords1.detach()
        flow = coords1 - coords0
        
        if add_noise:
            if noise_type == 'region':
                flow, noise = add_region_noise.add_noise(flow, self.noise_weight)
        
        flows_dense = tuple([flow]*self.split)

        flow_predictions = []

        if add_noise:
            resflows = tuple([torch.zeros_like(flow)]*self.split)

        for iter in range(diters):

            corr_map_list = []

            for i in range(self.split):
                coords = (coords0 + flows_dense[i]*((i+1)*0.2)).detach()
                corr_map = corr_fn_list[i](coords)
                corr_map_list.append(corr_map)

            corr_maps = torch.cat(corr_map_list, dim=0) 

            #=========================Independent prediction===============================
            # Treating the slices as the same batch
            mfs = self.mfe(torch.cat(flows_dense, dim=0), corr_maps)
            mfs = mfs.chunk((self.split), dim=0)
            mfs = self.mpa(mfs)
            net, dflow, upmask = self.dSupdate(net, inp, mfs)
            dflow = dflow.chunk(self.split, dim=0)
            # upmask = upmask.chunk(self.split, dim=0)
            flows_dense = tuple(flows_dense[i] + dflow[i] for i in range(self.split))
            


            if self.training:
                flows_dense_cat = torch.cat(flows_dense, dim=0)
                flows_dense_up_cat = self.upsample_flow(flows_dense_cat, upmask)
                flows = flows_dense_up_cat.chunk(5, dim=0)
                flows_scaled = torch.cat([flows[i]*((i+1)*0.2) for i in range(len(flows))], dim=0)
                flow_predictions.append(flows_scaled)
            
            if add_noise:
                resflows = tuple(resflows[i] + dflow[i] for i in range(self.split))
        
        if add_noise:
            # flow_result.update({'resflows':resflows})
            flow_result.update({'resflows':self.linear_upsample_flow(resflows[2])})
        
        
        if self.training:
            flow_result.update({'dense_pred_list':flow_predictions})
            return flow_result
        else:
            flows_dense_cat = torch.cat(flows_dense, dim=0) #[B*5, 2, H, W]
            flows_dense_up_cat = self.upsample_flow(flows_dense_cat, upmask)
            flows = flows_dense_up_cat.chunk(5, dim=0)
            flows_scaled = torch.cat([flows[i]*((i+1)*0.2) for i in range(len(flows))], dim=0)
            flow_result.update({'dense_pred':flows_scaled})
            return flow_result
