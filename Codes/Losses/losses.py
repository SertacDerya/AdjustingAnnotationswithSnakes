
import torch
from torch import nn
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as dist
from .gradRib import GradImRib, makeGaussEdgeFltr, cmptGradIm
import random

class MSELoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target, weights=None):
        loss = (pred-target).pow(2)
        if weights is not None:
            loss *= weights

        if self.ignore_index is not None:
            loss = loss[target!=self.ignore_index]

        return loss.mean()
    
class MAELoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target, weights=None):
        loss = torch.abs(pred-target)
        if weights is not None:
            loss *= weights

        if self.ignore_index is not None:
            loss = loss[target!=self.ignore_index]

        return loss.mean()

class SnakeFastLoss(nn.Module):
    def __init__(self, stepsz, alpha, beta, fltrstdev, ndims, nsteps, nsteps_width,
                 cropsz, dmax, maxedgelen, extgradfac, slow_start, negative_weight=1.5,
                 vis_seed=42, vis_sample_index=0):
        super(SnakeFastLoss, self).__init__()
        self.stepsz = stepsz
        self.alpha = alpha
        self.beta = beta
        self.fltrstdev = fltrstdev
        self.ndims = ndims
        self.cropsz = cropsz
        self.dmax = dmax
        self.maxedgelen = maxedgelen
        self.extgradfac = extgradfac
        self.nsteps = nsteps
        self.nsteps_width = nsteps_width
        self.slow_start = slow_start

        self.fltr = makeGaussEdgeFltr(self.fltrstdev, self.ndims)
        self.fltrt = torch.from_numpy(self.fltr).type(torch.float32)

        self.iscuda = False

        self.negative_weight = negative_weight
        
        # Enhancement factor to match dataset.py
        self.enhancement_factor = 10.0
        
        # NEW: Enable visualization with fixed seed
        self.visualize_maps = True  # Set to True to enable visualization
        self.vis_dir = 'snake_visualizations'
        self.vis_seed = vis_seed
        self.vis_sample_index = vis_sample_index
        
        # Set random seed for reproducible visualizations
        random.seed(self.vis_seed)
        np.random.seed(self.vis_seed)
        torch.manual_seed(self.vis_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.vis_seed)

    def cuda(self):
        super(SnakeFastLoss, self).cuda()
        self.fltrt = self.fltrt.cuda()
        self.iscuda = True
        return self

    def forward(self, pred_dmap, lbl_graphs, crops=None, mask= None, epoch=float("inf")):
        # pred_dmap is the predicted distance map from the UNet
        # lbl_graphs contains graphs each represent a label as a snake
        # crops is a list of slices, each represents the crop area of the corresponding snake

        pred_ = pred_dmap
        dmapW = torch.abs(pred_).clone()
        gimgW = cmptGradIm(dmapW, self.fltrt)
        gimg = cmptGradIm(pred_, self.fltrt)
        gimg *= self.extgradfac
        gimgW *= self.extgradfac
        snake_dmap = []

        for i, lg in enumerate(zip(lbl_graphs, gimg, gimgW)):
            # i is index num
            # lg is a tuple of a graph and a gradient image
            l = lg[0]  # graph
            g = lg[1]  # gradient image
            gw = lg[2]

            s = GradImRib(graph=l, crop=None, stepsz=self.stepsz, alpha=self.alpha,
                        beta=self.beta,dim=self.ndims, gimgV=g, gimgW=gw)
                    
            if self.iscuda: 
                s.cuda()
            if self.slow_start < epoch:
                s.optim(self.nsteps, self.nsteps_width)

            dmap = s.render_distance_map_with_widths(g[0].shape)
            if mask is not None:
                dmap = dmap * (mask==0)
            snake_dmap.append(dmap)

        snake_dm = torch.stack(snake_dmap, 0).unsqueeze(1) 

        snake_min, snake_max = snake_dm.min().item(), snake_dm.max().item()
        snake_range = snake_max - snake_min
        
        # If the range seems unenhanced (targets max around 16, min around -4)
        # then apply enhancement ourselves
        if snake_range < 30 and snake_min > -10:
            snake_negative_mask = snake_dm < 0
            snake_dm[snake_negative_mask] *= self.enhancement_factor
            enhanced = True
        else:
            enhanced = False
        
        # Store the enhanced snake_dm for visualization access
        self.snake_dm = snake_dm
            
        # DEBUG: Compare ranges every 50 epochs
        if epoch % 50 == 0:
            pred_min, pred_max = pred_dmap.min().item(), pred_dmap.max().item()
            snake_min, snake_max = snake_dm.min().item(), snake_dm.max().item()
            pred_neg_count = (pred_dmap < 0).sum().item()
            snake_neg_count = (snake_dm < 0).sum().item()
            pred_neg_percent = 100 * pred_neg_count / pred_dmap.numel()
            snake_neg_percent = 100 * snake_neg_count / snake_dm.numel()
            
            print(f"\n=== SnakeFastLoss Debug - Epoch {epoch} ===")
            print(f"Prediction range: {pred_min:.2f} to {pred_max:.2f} (range: {pred_max-pred_min:.2f})")
            print(f"Snake target range: {snake_min:.2f} to {snake_max:.2f} (range: {snake_max-snake_min:.2f})")
            print(f"Prediction negative pixels: {pred_neg_count} ({pred_neg_percent:.2f}%)")
            print(f"Snake target negative pixels: {snake_neg_count} ({snake_neg_percent:.2f}%)")
            print(f"Enhancement applied here: {enhanced}")
            print(f"Negative weighting factor: {self.negative_weight}")
            
            # Check if there's a range mismatch
            range_ratio = (pred_max - pred_min) / (snake_max - snake_min) if (snake_max - snake_min) > 0 else 0
            print(f"Prediction/Snake range ratio: {range_ratio:.2f}")
            
            if snake_neg_count == 0:
                print("⚠️  WARNING: Snake targets have NO negative values!")
            if pred_neg_count == 0:
                print("⚠️  WARNING: Predictions have NO negative values!")
            
            # NEW: Visualize the maps with fixed sample index
            if self.visualize_maps:
                try:
                    self.visualize_distance_maps(pred_dmap, snake_dm, epoch, sample_idx=self.vis_sample_index)
                except Exception as e:
                    print(f"Warning: Could not create visualization: {str(e)}")
        
        # MODIFIED LOSS CALCULATION:
        # Calculate MSE loss
        squared_diff = (pred_dmap - snake_dm)**2
        
        # Apply moderate weighting to negative target values
        negative_mask = (snake_dm < 0).float()
        weighted_squared_diff = squared_diff * (1.0 + negative_mask * (self.negative_weight - 1.0))

        # Calculate the final loss
        loss = weighted_squared_diff.mean()
        
        # Debug: Show loss components every 50 epochs
        if epoch % 50 == 0:
            print(f"Loss: {loss.item():.4f}")
            print("=" * 50)
        
        self.snake = s
        return loss


    
class SnakeSimpleLoss(nn.Module):
    def __init__(self, stepsz,alpha,beta,fltrstdev,ndims,nsteps,
                       cropsz,dmax,maxedgelen,extgradfac):
        super(SnakeSimpleLoss,self).__init__()
        self.stepsz=stepsz
        self.alpha=alpha
        self.beta=beta
        self.fltrstdev=fltrstdev
        self.ndims=ndims
        self.cropsz=cropsz
        self.dmax=dmax
        self.maxedgelen=maxedgelen
        self.extgradfac=extgradfac
        self.nsteps=nsteps

        self.fltr =makeGaussEdgeFltr(self.fltrstdev,self.ndims)
        self.fltrt=torch.from_numpy(self.fltr).type(torch.float32)

        self.iscuda=False

    def cuda(self):
        super(SnakeSimpleLoss,self).cuda()
        self.fltrt=self.fltrt.cuda()
        self.iscuda=True
        return self

    def forward(self,pred_dmap,lbl_graphs,crops=None):
    
        pred_=pred_dmap.detach()
        dmapW = torch.abs(pred_).clone()
        gimgW = cmptGradIm(dmapW.detach(), self.fltrt)
        gimg = cmptGradIm(pred_.detach(), self.fltrt)
        gimg *= self.extgradfac
        gimgW *= self.extgradfac
        snake_dmap = []

        for i,lg in enumerate(zip(lbl_graphs,gimg, gimgW)):
            l = lg[0]
            g = lg[1]
            gw = lg[2]
 
            s = GradImRib(graph=l, crop=None, stepsz=self.stepsz, alpha=self.alpha,
                        beta=self.beta,dim=self.ndims, gimgV=g, gimgW=gw)
            
            if self.iscuda: 
                s.cuda()

            s.optim(self.nsteps)

            dmap = s.render_distance_map_with_widths(g[0].shape)
            snake_dmap.append(dmap)

        snake_dm=torch.stack(snake_dmap,0).unsqueeze(1)
        loss=torch.pow(pred_dmap-snake_dm,2).mean()
                  
        self.snake=s
        self.gimg=gimg
        
        return loss
    
    
