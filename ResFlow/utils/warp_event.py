"""
This function if for warp the events

Warp_Events: 
    give an events array [n, 4] and flow [2, H, W], return the e-fwl

    events ... [n, 4]
    flow ... [2, H, W]
    flows ... (5), [2, H, W]
    IWE ... [H, W]

"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
import imageio
from PIL import Image
from .flow_viz import flow_to_image
import cv2

import matplotlib.pyplot as plt


class WarpEventsEraftInterp(object):
    """
    Warp functions class.
    """

    def __init__(self,image_size=(480,640), vis_IWE=False, vis_Flow=False):
        self.image_size = image_size
        self.bins=15
        
        self.vis_IWE = vis_IWE
        self.vis_dir = './VISCHECK/IWE/'
        if self.vis_IWE:
            os.makedirs(self.vis_dir, exist_ok=True)

        self.vis_Flow = vis_Flow
        self.flow_vis_dir = './VISCHECK/FLOW/'
        if self.vis_Flow:
            os.makedirs(self.flow_vis_dir, exist_ok=True)
        
            
    def get_IWE(self, events, sigma=-1):
        """Create IWE for events array.

        Inputs:
            events ... [n, 4], (x, y, t, p).
            sigma ... Sigma for the gaussian blur.
        Returns:
            IWE ... [H, W].
        """

        # assert torch.is_tensor(events)
        
        
        IWE = self.bilinear_vote_tensor(events)
        
        if sigma > 0:
            IWE = gaussian_filter(IWE, sigma)
        return IWE
    


    def bilinear_vote_tensor(self, events, weight=1.0):
        """Tensor version of `bilinear_vote_numpy().`
        """
        if type(weight) == torch.Tensor:
            assert weight.shape == events.shape[:-1]
        if len(events.shape) == 2:
            events = events[None, ...]  # 1 x n x 4

        h, w = self.image_size
        nb = len(events)
        image = events.new_zeros((nb, h * w))

        floor_xy = torch.floor(events[..., :2] + 1e-6)
        floor_to_xy = events[..., :2] - floor_xy
        floor_xy = floor_xy.long()

        x1 = floor_xy[..., 1]
        y1 = floor_xy[..., 0]
        inds = torch.cat(
            [
                x1 + y1 * w,
                x1 + (y1 + 1) * w,
                (x1 + 1) + y1 * w,
                (x1 + 1) + (y1 + 1) * w,
            ],
            dim=-1,
        )  # [(b, ) n_events x 4]
        inds_mask = torch.cat(
            [
                (0 <= x1) * (x1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1) * (x1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
            ],
            axis=-1,
        )

        w_pos0 = (1 - floor_to_xy[..., 0]) * (1 - floor_to_xy[..., 1]) * weight
        w_pos1 = floor_to_xy[..., 0] * (1 - floor_to_xy[..., 1]) * weight
        w_pos2 = (1 - floor_to_xy[..., 0]) * floor_to_xy[..., 1] * weight
        w_pos3 = floor_to_xy[..., 0] * floor_to_xy[..., 1] * weight
        vals = torch.cat([w_pos0, w_pos1, w_pos2, w_pos3], dim=-1)  # [(b,) n_events x 4]

        inds = (inds * inds_mask).long()
        vals = vals * inds_mask
        image.scatter_add_(1, inds, vals)
        return image.reshape((nb,) + self.image_size).squeeze()
    
    def grid_sample_values(self, input, height, width):
        # ================================ Grid Sample Values ============================= #
        # Input:    Torch Tensor [3,H*W]m where the 3 Dimensions mean [x,y,z]               #
        # Height:   Image Height                                                            #
        # Width:    Image Width                                                             #
        # --------------------------------------------------------------------------------- #
        # Output:   tuple(value_ipl, valid_mask)                                            #
        #               value_ipl       -> [H,W]: Interpolated values                       #
        #               valid_mask      -> [H,W]: 1: Point is valid, 0: Point is invalid    #
        # ================================================================================= #
        device = input.device
        ceil = torch.stack([torch.ceil(input[0,:]), torch.ceil(input[1,:]), input[2,:]])
        floor = torch.stack([torch.floor(input[0,:]), torch.floor(input[1,:]), input[2,:]])
        z = input[2,:].clone()

        values_ipl = torch.zeros(height*width, device=device)
        weights_acc = torch.zeros(height*width, device=device)
        # Iterate over all ceil/floor points
        for x_vals in [floor[0], ceil[0]]:
            for y_vals in [floor[1], ceil[1]]:
                # Mask Points that are in the image
                in_bounds_mask = (x_vals < width) & (x_vals >=0) & (y_vals < height) & (y_vals >= 0)

                # Calculate weights, according to their real distance to the floored/ceiled value
                weights = (1 - (input[0]-x_vals).abs()) * (1 - (input[1]-y_vals).abs())

                # Put them into the right grid
                indices = (x_vals + width * y_vals).long()
                values_ipl.put_(indices[in_bounds_mask], (z * weights)[in_bounds_mask], accumulate=True)
                weights_acc.put_(indices[in_bounds_mask], weights[in_bounds_mask], accumulate=True)

        # Mask of valid pixels -> Everywhere where we have an interpolated value
        valid_mask = weights_acc.clone()
        valid_mask[valid_mask > 0] = 1
        valid_mask= valid_mask.bool().reshape([height,width])

        # Divide by weights to get interpolated values
        values_ipl = values_ipl / (weights_acc + 1e-15)
        values_rs = values_ipl.reshape([height,width])

        return values_rs.unsqueeze(0).clone(), valid_mask.unsqueeze(0).clone()
    
    def Interp_forward_dflows(self, dflows):
        """ convert forward flow to backward flow
        Input:
            dflows: ... [(15), 2, H, W]
        
        Output:
            bflows: ... [(15), 2, H, W]
        """

        flow = torch.cat(dflows, dim=0)
        assert len(flow.shape) == 4

        b, _, h, w = flow.shape
        device = flow.device

        dx ,dy = flow[:,0], flow[:,1]
        y0, x0 = torch.meshgrid(torch.arange(0, h, 1), torch.arange(0, w, 1))
        x0 = torch.stack([x0]*b).to(device)
        y0 = torch.stack([y0]*b).to(device)

        x1 = x0 + dx
        y1 = y0 + dy

        x1 = x1.flatten(start_dim=1)
        y1 = y1.flatten(start_dim=1)
        dx = dx.flatten(start_dim=1)
        dy = dy.flatten(start_dim=1)

        flow_new = torch.zeros(flow.shape, device=device)
        for i in range(b):
            flow_new[i,0] = self.grid_sample_values(torch.stack([x1[i],y1[i],dx[i]]), h, w)[0]
            flow_new[i,1] = self.grid_sample_values(torch.stack([x1[i],y1[i],dy[i]]), h, w)[0]

        return list(flow_new.chunk(15, dim=0))
    

    def warp_events_from_dflows(self, events, dflows):
        """
        given the dflows between t0 and t1, warp the events

        events ... [n, 4] (x, y, t, p)
        dflows ... [(N), 2, H, W]

        warped_events ... [n, 4]
        """

        # insert a zeros tensor to dflows
        d = len(dflows) # 15

        dflows.insert(0, torch.zeros_like(dflows[0])) # [(16), 2, H, W]
        dflows = torch.cat(dflows, dim=0) # [16, 2, H, W]
        

        t = events[..., 2]
        norm_t = (t - t[0]) * d * 0.999 / (t[-1] - t[0])  # 0--14.99
        t_index = torch.floor(norm_t).long() # 0, 1, ..., 14
        t_delta = norm_t - t_index # 0--0.99
        
        # print(t_delta.mean(),t_delta.max())

        assert t_index.min() == 0
        assert t_index.max() == d-1

        x0 = events[..., 0].long()
        y0 = events[..., 1].long()
        

        disp_x = dflows[t_index, 1, x0, y0] + \
                 (dflows[t_index+1, 1, x0, y0] - dflows[t_index,1,x0,y0])*t_delta
        disp_y = dflows[t_index, 0, x0, y0] + \
                 (dflows[t_index+1, 0, x0, y0] - dflows[t_index,0,x0,y0])*t_delta

        new_x = x0 - disp_x
        new_y = y0 - disp_y

        warped_events = torch.stack((new_x, new_y, events[..., 2], events[..., 3]), dim=1)

        return warped_events
    
    def get_dflows_from_flow(self, flow):
        """
        Calculate dense flow from one flow in linear manner.
        
        Args:
            flow ... [1, 2, H, W]
            dflows ... (15), [1, 2, H, W]
        """

        du = flow / self.bins
        
        dflows = [du * i for i in range(1, self.bins+1)]
        
        return dflows
    
    def get_dflows_from_dflows(self, flows):
        """generate dense flows from multi flows

        Args:
            flows ... (5) [1, 2, H, W]
        Returns:
            dflows ... (15), [1, 2, H, W]
        """
        zflow = torch.zeros_like(flows[0])

        dflows = []

        for findex in range(0, len(flows)):
            
            if findex == 0:
                du = (flows[findex] - zflow)/3
                dflows.append(zflow + du * 1)
                dflows.append(zflow + du * 2)
                dflows.append(flows[findex])
            else:
                du = (flows[findex] - flows[findex-1])/3
                dflows.append(flows[findex -1] + du * 1)
                dflows.append(flows[findex -1] + du * 2)
                dflows.append(flows[findex])
        
        assert len(dflows) == self.bins
        
        return dflows
    
    
    def events_filter(self, events):
        x = events[:,0]
        y = events[:,1]
        # print("event[:,0] max min, events[:,1] max min", x.max(), x.min(), y.max(), y.min())
        mask = (x>0) * (x<self.image_size[1]) * (y>0) * (y<self.image_size[0])
        events = events[mask,:]
        events = torch.stack((events[:,1], events[:,0], events[:,2], events[:,3]), dim=1)
        return events

        
    def calculate_fwl_loss(self, events, flows, name=None, freq=5):
        """Calculate FWL
        
        Args:
            events ... [n, 4]
            flows ... [1, 2, H, W] or [(5), 1, 2, H, W].
            Convert flows to dense-flows(dflows), and then warp events

        Returns:
            fwl: flow error 1.
            rfwl: flow error 2.
        """
        events = torch.from_numpy(events).cuda()
        # print("device:", events.device, flows[0].device)
        # if len(flows.shape)==4 and flows.shape[0]==1:
        #     flows = flows.squeeze(0).cpu().numpy()

        # convert flows to dflows
        if freq == 5:
            if isinstance(flows, list):
                f_dflows = self.get_dflows_from_dflows(flows)
            else:
                f_dflows = self.get_dflows_from_flow(flows)
        else:
            f_dflows = flows
              
        dflows = self.Interp_forward_dflows(f_dflows)
        # dflows = f_dflows
        
        # cut events out of bound
        events = self.events_filter(events)
        # vote events to raw_IWE
        raw_IWE = self.get_IWE(events)
        
        # warp events by dflows
        warped_events = self.warp_events_from_dflows(events, dflows)
        # vote warped events to warped_IWE
        warped_IWE = self.get_IWE(warped_events)
        
        if self.vis_IWE and name is not None:
            self.vis_check_iwe(warped_IWE, f'{name}_1_war_linear')
            # self.vis_check_iwe(raw_IWE, f'{name}_0_raw')
            # self.vis_check_iwe_pol(warped_events, f'{name}_war_dense_pol')
            
        if self.vis_Flow and name is not None:
            self.vis_check_nl_flow(f_dflows, f'{name}_nl_flow')
            self.vis_check_flow(f_dflows, f'{name}_flow')
        
        # if self.vis_Flow and name is not None:
        #     self.vis_check_flow(dflows, f'{name}_flow')
        
        # e-fwl
        fwl = torch.var(warped_IWE) / (torch.var(raw_IWE) + 0.0001)  
        
        # fwl = torch.var(warped_IWE/torch.count_nonzero(warped_IWE)) / (torch.var(raw_IWE/torch.count_nonzero(raw_IWE)) + 0.00000000000001) 
        
        
        return fwl
    
#     def vis_check_iwe_pol(self, events, name, threshold=0.5):
#         # vote events
#         IWE = self.bilinear_vote_tensor(events, events[:,3]*2-1)
#         IWE = IWE.cpu().numpy()
#         img = np.zeros((self.image_size[0], self.image_size[1], 3))

#         # 根据矩阵元素的值设置颜色
#         img[IWE > threshold] = [255, 0, 0]  # 正值显示为红色
#         img[IWE < -threshold] = [0, 0, 255]  # 负值显示为蓝色
        
#         IWE_weight = (np.abs(IWE)/np.max(np.abs(IWE))+0.8)/1.68
#         img = img * IWE_weight[:,:,None]
        
#         img[np.abs(IWE) <= threshold] = [255, 255, 255]  # 小于阈值的值显示为白色
        
#         img_IWE = Image.fromarray(img.astype(np.uint8))
#         img_pth=os.path.join(self.vis_dir,f'{name}_pol.png')
#         img_IWE.save(img_pth) 
        
    def vis_check_iwe_pol(self, events, name, threshold=1):
        # vote events
        IWE = self.bilinear_vote_tensor(events, events[:,3]*2-1)
        IWE = IWE.cpu().numpy()
        
        # IWE[IWE>10]=10
        # IWE[IWE<-10]=-10
        
        img = np.zeros((self.image_size[0], self.image_size[1], 3))
        imgw = np.ones((self.image_size[0], self.image_size[1], 3))*255

        # 根据矩阵元素的值设置颜色
        img[IWE > threshold] = [255, 0, 0]  # 正值显示为红色
        img[IWE < -threshold] = [0, 0, 255]  # 负值显示为蓝色
        
        IWE_weight =  (np.abs(IWE)/np.max(np.abs(IWE))+0.01)
        IWE_weight = np.sqrt(np.sqrt(IWE_weight))
        img = img * IWE_weight[:,:,None] + imgw * (1 - IWE_weight[:,:,None])
        
        img[np.abs(IWE) <= threshold] = [255, 255, 255]  # 小于阈值的值显示为白色
        
        img_IWE = Image.fromarray(img.astype(np.uint8))
        img_pth=os.path.join(self.vis_dir,f'{name}_pol.png')
        img_IWE.save(img_pth)
    
    def vis_check_iwe(self, IWE, name):
        
        H, W = IWE.shape
        
        norm_IWE = 1- (IWE - IWE.min()) / (IWE.max()-IWE.min())
        img_IWE = Image.fromarray((norm_IWE.cpu().numpy() * 255).astype(np.uint8))
        img_pth=os.path.join(self.vis_dir,f'{name}_event.png')
        img_IWE.save(img_pth)  
        
    def vis_check_flow(self, dflows, name):
        """save the flows to check

        Args:
            dflows ... (15), [B, 2, H, W]
            name ... str
        """
        for i in [2,5,8,11,14]:
            flow = dflows[i]#[ ( i + 1 ) * 3 - 1 - 1 ]
            flow = flow[0, ...].squeeze().permute(1,2,0)
            flow_img = flow_to_image(flow.cpu().numpy())
            
            # flow_img = self.draw_arrow_on_flow(flow_img, flow, i+1)
            
            flow_pth = os.path.join(self.flow_vis_dir, f'{name}_image_{i}.png')
            imageio.imwrite(flow_pth, flow_img)
    
    def draw_arrow_on_flow(self, flow_img, flow, scale):
        H, W, _ = flow_img.shape
        arrow_image = flow_img.copy()
        
        step = 8

        # 遍历图像，在每个步长位置绘制箭头
        for y in range(0, H, step):
            for x in range(0, W, step):
                # 获取光流的方向和大小
                dx, dy = flow[y, x]

                # 绘制箭头
                end_x = int(x + dx * (15/scale))
                end_y = int(y + dy * (15/scale))
                cv2.arrowedLine(arrow_image, (x, y), (end_x, end_y), color=(0, 0, 0), thickness=1, tipLength=0.3)
        
        return arrow_image
            
    def vis_check_nl_flow(self, dflows, name):
        """ draw flows on image
        Args:
            dflows ... [(15), B, 2, H, W]
            name ... str
        """
        
        dflows.insert(0, torch.zeros_like(dflows[0]))
        
        Img = np.zeros((self.image_size[0], self.image_size[1], 3)).astype(np.uint8)
        
        for i in range(len(dflows)):
            Img = self.draw_point_on_flow(Img, dflows[i], i)
        
        flow_path = os.path.join(self.flow_vis_dir, f'{name}_nl_flow.png')
        imageio.imwrite(flow_path, Img)
    
    def draw_point_on_flow(self, Img, flow, index):
        """ draw point on image
        Args:
            flow ... [B, 2, H, W]
            Img ... [H, W, 3]
        """
        H, W, _ = Img.shape
        point_Img = Img.copy()
        
        flow = flow[0, ...].squeeze().permute(1,2,0)
        
        step = 32
        
        # 选择一个 colormap
        cmap = plt.get_cmap('YlOrRd', 16)

        # 获取 16 种颜色
        colors = [cmap(i)[:3] for i in range(16)]  # 获取 RGB 值

        # 将颜色转换为 0-255 范围的整数
        colors_255 = [[int(c * 255) for c in color] for color in colors]
        
        # 遍历图像，在每个步长位置绘制箭头
        for y in range(0, H, step):
            for x in range(0, W, step):
                # 获取光流的方向和大小
                dx, dy = flow[y, x]

                end_x = int(x + dx)
                end_y = int(y + dy)

                radius = 1  # 点的半径
                color = colors_255[index]  # 点的颜色，这里使用绿色
                thickness = -1  # 如果thickness为-1，则填充整个圆

                cv2.circle(point_Img, (end_x, end_y), radius, color, thickness)
                # cv2.drawMarker(point_Img, (end_x, end_y), color, markerType=cv2.MARKER_CROSS, markerSize=1)
        
        return point_Img




class WarpEventsDenseFlows(object):
    """
    Warp functions class.
    """

    def __init__(self,image_size=(480,640), vis_IWE=False, vis_Flow=False):
        self.image_size = image_size
        self.bins=15
        
        self.vis_IWE = vis_IWE
        self.vis_dir = './VISCHECK/IWE/'
        if self.vis_IWE:
            os.makedirs(self.vis_dir, exist_ok=True)
            
        self.vis_Flow = vis_Flow
        self.flow_vis_dir = './VISCHECK/Flow/'
        if self.vis_Flow:
            os.makedirs(self.flow_vis_dir, exist_ok=True)
        
        
            
    def get_IWE(self, events, sigma=-1):
        """Create IWE for events array.

        Inputs:
            events ... [n, 4], (x, y, t, p).
            sigma ... Sigma for the gaussian blur.
        Returns:
            IWE ... [H, W].
        """

        # assert torch.is_tensor(events)
        
        
        IWE = self.bilinear_vote_tensor(events)
        
        if sigma > 0:
            IWE = gaussian_filter(IWE, sigma)
        return IWE
    


    def bilinear_vote_tensor(self, events, weight=1.0):
        """Tensor version of `bilinear_vote_numpy().`
        """
        
        if len(events.shape) == 2:
            events = events[None, ...]  # 1 x n x 4

        h, w = self.image_size
        nb = len(events)
        image = events.new_zeros((nb, h * w))

        floor_xy = torch.floor(events[..., :2] + 1e-6)
        floor_to_xy = events[..., :2] - floor_xy
        floor_xy = floor_xy.long()

        x1 = floor_xy[..., 1]
        y1 = floor_xy[..., 0]
        inds = torch.cat(
            [
                x1 + y1 * w,
                x1 + (y1 + 1) * w,
                (x1 + 1) + y1 * w,
                (x1 + 1) + (y1 + 1) * w,
            ],
            dim=-1,
        )  # [(b, ) n_events x 4]
        inds_mask = torch.cat(
            [
                (0 <= x1) * (x1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1) * (x1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
            ],
            axis=-1,
        )

        w_pos0 = (1 - floor_to_xy[..., 0]) * (1 - floor_to_xy[..., 1]) * weight
        w_pos1 = floor_to_xy[..., 0] * (1 - floor_to_xy[..., 1]) * weight
        w_pos2 = (1 - floor_to_xy[..., 0]) * floor_to_xy[..., 1] * weight
        w_pos3 = floor_to_xy[..., 0] * floor_to_xy[..., 1] * weight
        vals = torch.cat([w_pos0, w_pos1, w_pos2, w_pos3], dim=-1)  # [(b,) n_events x 4]

        inds = (inds * inds_mask).long()
        vals = vals * inds_mask
        image.scatter_add_(1, inds, vals)
        return image.reshape((nb,) + self.image_size).squeeze()
    

    def warp_events_from_dflows(self, events, dflows):
        """
        given the dflows between t0 and t1, warp the events

        events ... [n, 4] (x, y, t, p)
        dflows ... [(N), 2, H, W]

        warped_events ... [n, 4]
        """

        # insert a zeros tensor to dflows
        d = len(dflows) # 15

        dflows.insert(0, torch.zeros_like(dflows[0])) # [(16), 2, H, W]
        dflows = torch.cat(dflows, dim=0) # [16, 2, H, W]
        

        t = events[..., 2]
        norm_t = (t - t[0]) * d * 0.999 / (t[-1] - t[0])  # 0--14.99
        t_index = torch.floor(norm_t).long() # 0, 1, ..., 14
        t_delta = norm_t - t_index # 0--0.99
        
        # print(t_delta.mean(),t_delta.max())

        assert t_index.min() == 0
        assert t_index.max() == d-1

        x0 = events[..., 0].long()
        y0 = events[..., 1].long()
        

        disp_x = dflows[t_index, 1, x0, y0] + \
                 (dflows[t_index+1, 1, x0, y0] - dflows[t_index,1,x0,y0])*t_delta
        disp_y = dflows[t_index, 0, x0, y0] + \
                 (dflows[t_index+1, 0, x0, y0] - dflows[t_index,0,x0,y0])*t_delta

        new_x = x0 - disp_x
        new_y = y0 - disp_y

        warped_events = torch.stack((new_x, new_y, events[..., 2], events[..., 3]), dim=1)

        return warped_events
    
    def get_dflows_from_flow(self, flow):
        """
        Calculate dense flow from one flow in linear manner.
        
        Args:
            flow ... [1, 2, H, W]
            dflows ... (15), [1, 2, H, W]
        """

        du = flow / self.bins
        
        dflows = [du * i for i in range(1, self.bins+1)]
        
        return dflows
    
    def get_dflows_from_dflows(self, flows):
        """generate dense flows from multi flows

        Args:
            flows ... (5) [1, 2, H, W]
        Returns:
            dflows ... (15), [1, 2, H, W]
        """
        zflow = torch.zeros_like(flows[0])

        dflows = []

        for findex in range(0, len(flows)):
            
            if findex == 0:
                du = (flows[findex] - zflow)/3
                dflows.append(zflow + du * 1)
                dflows.append(zflow + du * 2)
                dflows.append(flows[findex])
            else:
                du = (flows[findex] - flows[findex-1])/3
                dflows.append(flows[findex -1] + du * 1)
                dflows.append(flows[findex -1] + du * 2)
                dflows.append(flows[findex])
        
        assert len(dflows) == self.bins
        
        return dflows
    
    
    def events_filter(self, events):
        x = events[:,0]
        y = events[:,1]
        # print("event[:,0] max min, events[:,1] max min", x.max(), x.min(), y.max(), y.min())
        mask = (x>0) * (x<self.image_size[1]) * (y>0) * (y<self.image_size[0])
        events = events[mask,:]
        events = torch.stack((events[:,1], events[:,0], events[:,2], events[:,3]), dim=1)
        return events

        
    def calculate_fwl_loss(self, events, flows, name=None):
        """Calculate FWL
        
        Args:
            events ... [n, 4]
            flows ... [1, 2, H, W] or [(5), 1, 2, H, W].
            Convert flows to dense-flows(dflows), and then warp events

        Returns:
            fwl: flow error 1.
            rfwl: flow error 2.
        """
        events = torch.from_numpy(events).cuda()
        # print("device:", events.device, flows[0].device)
        # if len(flows.shape)==4 and flows.shape[0]==1:
        #     flows = flows.squeeze(0).cpu().numpy()

        # convert flows to dflows
        if isinstance(flows, list):
            dflows = self.get_dflows_from_dflows(flows)
        else:
            dflows = self.get_dflows_from_flow(flows)
        
        # cut events out of bound
        events = self.events_filter(events)
        # vote events to raw_IWE
        raw_IWE = self.get_IWE(events)
        
        # warp events by dflows
        warped_events = self.warp_events_from_dflows(events, dflows)
        # vote warped events to warped_IWE
        warped_IWE = self.get_IWE(warped_events)
        
        if self.vis_IWE and name is not None:
            self.vis_check_iwe(warped_IWE, f'{name}_war')
            self.vis_check_iwe(raw_IWE, f'{name}_raw')
        
        if self.vis_Flow and name is not None:
            # self.vis_check_nl_flow(f_dflows, f'{name}_flow')
            self.vis_check_flow(dflows, f'{name}_flow')
        
        # e-fwl
        fwl = torch.var(warped_IWE) / (torch.var(raw_IWE) + 0.01)  
        
        return fwl
    
    def vis_check_flow(self, dflows, name):
        """save the flows to check

        Args:
            dflows ... (15), [B, 2, H, W]
            name ... str
        """
        for i in [2,5,8,11,14]:
            flow = dflows[i]#[ ( i + 1 ) * 3 - 1 - 1 ]
            flow = flow[0, ...].squeeze().permute(1,2,0)
            flow_img = flow_to_image(flow.cpu().numpy())
            
            # flow_img = self.draw_arrow_on_flow(flow_img, flow, i+1)
            
            flow_pth = os.path.join(self.flow_vis_dir, f'{name}_image_{i}.png')
            imageio.imwrite(flow_pth, flow_img)
    
    def vis_check_iwe(self, IWE, name):
        
        H, W = IWE.shape
        
        norm_IWE = (IWE - IWE.min()) / (IWE.max()-IWE.min())
        img_IWE = Image.fromarray((norm_IWE.cpu().numpy() * 255).astype(np.uint8))
        img_pth=os.path.join(self.vis_dir,f'{name}_event.png')
        img_IWE.save(img_pth)  


class WarpEvents(object):
    """
    Warp functions class.
    """

    def __init__(self,image_size=(480,640), vis_IWE=False):
        self.image_size = image_size
        self.vis_IWE = vis_IWE
        self.vis_dir = './VISCHECK/E_IWE/'
        if self.vis_IWE:
            os.makedirs(self.vis_dir, exist_ok=True)
        
            
    def get_IWE(self, events, sigma=0.):
        """Create IWE for events array.

        Inputs:
            events ... [n, 4], (x, y, t, p).
            sigma ... Sigma for the gaussian blur.
        Returns:
            IWE ... [H, W].
        """
        
        IWE = self.bilinear_vote_numpy(events)
        
        if sigma > 0:
            IWE = gaussian_filter(IWE, sigma)
        return IWE
    
    def bilinear_vote_numpy(self, events, weight=1.):
        """Use bilinear voting to get IWE.

        Args:
            events ... [n, 4], (x, y, t, p). x is the height(0, 480)
            weight (float or np.ndarray) ... Weight to multiply to the voting value.
                If scalar, the weight is all the same among events.
                If it's array-like, it should be the shape of [n_events].
                Defaults to 1.0.

        Returns:
            image ... [1, H, W].
        """
        
        if len(events.shape) == 2:
            events = events[None, ...]  # 1 x n x 4

        # x-y is height-width
        h, w = self.image_size
        nb = len(events)
        image = np.zeros((nb, h * w))

        floor_xy = np.floor(events[..., :2] + 1e-8)     # lower
        floor_to_xy = events[..., :2] - floor_xy        # int

        x1 = floor_xy[..., 1]
        y1 = floor_xy[..., 0]
        
        # print("cnmdbbbbbbbbb",x1.shape, y1.shape)
        inds = np.concatenate(
            [
                x1 + y1 * w,
                x1 + (y1 + 1) * w,
                (x1 + 1) + y1 * w,
                (x1 + 1) + (y1 + 1) * w,
            ],
            axis=-1,
        )
        inds_mask = np.concatenate(
            [
                (0 <= x1) * (x1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1) * (x1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
            ],
            axis=-1,
        )
        w_pos0 = (1 - floor_to_xy[..., 0]) * (1 - floor_to_xy[..., 1]) * weight
        w_pos1 = floor_to_xy[..., 0] * (1 - floor_to_xy[..., 1]) * weight
        w_pos2 = (1 - floor_to_xy[..., 0]) * floor_to_xy[..., 1] * weight
        w_pos3 = floor_to_xy[..., 0] * floor_to_xy[..., 1] * weight
        vals = np.concatenate([w_pos0, w_pos1, w_pos2, w_pos3], axis=-1)
        inds = (inds * inds_mask).astype(np.int64)
        vals = vals * inds_mask
        for i in range(nb):
            np.add.at(image[i], inds[i], vals[i])
        return image.reshape((nb,) + self.image_size).squeeze()
    

    def warp_events_from_one_flow(self, events, flow):
        """
        given the flow between t0 and t1, warp the events

        events ... [n, 4] (x, y, t, p)
        flow ... [2, H, W]

        warped_events ... [n, 4]
        """

        t = events[..., 2]
        norm_t = (t - t[0]) / (t[-1] - t[0])

        x0 = events[..., 0].astype(np.int32)
        y0 = events[..., 1].astype(np.int32)
        
        disp_x = flow[1, x0, y0] * norm_t    # the displace of x
        disp_y = flow[0, x0, y0] * norm_t   # the displace of y

        new_x = x0 - disp_x
        new_y = y0 - disp_y

        warped_events = np.stack((new_x, new_y, events[..., 2], events[..., 3]), axis=1)

        return warped_events
    
    
    def events_filter(self, events):
        x = events[:,0]
        y = events[:,1]
        
        # print("event[:,0] max min, events[:,1] max min", x.max(), x.min(), y.max(), y.min())
        mask = (x>0) * (x<self.image_size[1]) * (y>0) * (y<self.image_size[0])
        events = events[mask,:]
        events = np.stack((events[:,1], events[:,0], events[:,2], events[:,3]), axis=1)
        return events

        
    def calculate_fwl_loss(self, events, flows):
        """Calculate FWL
        
        Args:
            events ... [n, 4]
            flows ... [1, 2, H, W].
            warp_type ... 1 or 5

        Returns:
            fwl: flow error 1.
            rfwl: flow error 2.
        """
        # if len(flows.shape)==4 and flows.shape[0]==1:
        #     flows = flows.squeeze(0).cpu().numpy()
        
        events = self.events_filter(events)
        
        raw_IWE = self.get_IWE(events)
        warped_events = self.warp_events_from_one_flow(events, flows)
        warped_IWE = self.get_IWE(warped_events)
        
        if self.vis_IWE:
            self.vis_check_iwe(warped_IWE, 'war')
        
        # e-fwl
        fwl = np.var(warped_IWE) / np.var(raw_IWE)  
        
        return fwl
    
    def vis_check_iwe(self, IWE, name):
        H, W = IWE.shape
        
        norm_IWE = 1- (IWE - IWE.min()) / (IWE.max()-IWE.min())
        img_IWE = Image.fromarray((norm_IWE * 255).astype(np.uint8))
        img_pth=os.path.join(self.vis_dir,f'{name}_event.png')
        img_IWE.save(img_pth)


################################
    

    
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



#################################    
if __name__ == "__main__":
    
    events_file = '/data/zqa1313/DSEC-Flow/DSEC_Events/test/zurich_city_14_c/001140.npz'
    
    events = np.load(events_file)
    # print(events)
    print(events['events_prev'].shape)
    
    events = events['events_prev']
    
    print("*********",events[:,:2].min())
    
    flow_file = '/data/zqa1313/DSEC-Flow/STFlow-0663/upload_200k/zurich_city_14_c/001140.png'
    
    flow_16bit = imageio.imread(flow_file, format='PNG-FI')
    
    flow_map, valid = flow_16bit_to_float(flow_16bit)
    
    flow_map = flow_map.transpose(2, 0, 1)
    
    print(flow_map.shape)
    
    warp1 = WarpEvents(vis_IWE=True)
    
    e_fwl = warp1.calculate_fwl_loss(events, flow_map)
    
    print(e_fwl)
    
    
    