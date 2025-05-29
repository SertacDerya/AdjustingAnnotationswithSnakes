
import torch
from torch import nn
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as dist
from .gradRib import GradImRib, makeGaussEdgeFltr, cmptGradIm
import random
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

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

def visualize_distance_maps(self, pred_dmap, snake_dm, epoch, sample_idx=None):
        """
        Visualize the predicted distance map and snake target map.
        
        Args:
            pred_dmap: Predicted distance map tensor from the network
            snake_dm: Snake distance map tensor (ground truth)
            epoch: Current epoch number (for filenames)
            sample_idx: Index of the sample in the batch to visualize
        """
        # Set fixed random seed for consistent visualization
        random.seed(self.vis_seed)
        np.random.seed(self.vis_seed)
        torch.manual_seed(self.vis_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.vis_seed)
        
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Use the class-specified sample index if none provided
        if sample_idx is None:
            sample_idx = self.vis_sample_index
            
        print(f"Using consistent visualization: sample_idx={sample_idx}, seed={self.vis_seed}")
        
        # Debug dimensions before processing
        print(f"DEBUG: pred_dmap shape: {pred_dmap.shape}")
        print(f"DEBUG: snake_dm shape: {snake_dm.shape}")
        
        # Helper function to extract 2D slice from tensor of any dimension
        def extract_2d_slice(tensor, idx=0):
            # Convert to numpy
            tensor_np = tensor.detach().cpu().numpy()
            
            # Keep reducing dimensions until we get a 2D array
            while tensor_np.ndim > 2:
                # If first dimension is larger than 1, take the specified index
                if tensor_np.shape[0] > 1:
                    if idx < tensor_np.shape[0]:
                        tensor_np = tensor_np[idx]
                    else:
                        print(f"Warning: Index {idx} out of bounds for dimension of size {tensor_np.shape[0]}. Using index 0.")
                        tensor_np = tensor_np[0]
                # Otherwise just squeeze the first dimension
                else:
                    tensor_np = tensor_np.squeeze(0)
                
                # Print the new shape for debugging
                print(f"After reduction: shape = {tensor_np.shape}")
                
                # Safety check - if we've reduced too much, stop
                if tensor_np.ndim <= 2:
                    break
            
            # Final squeeze to handle any remaining singleton dimensions
            tensor_np = np.squeeze(tensor_np)
            
            # Ensure we have a 2D array
            if tensor_np.ndim != 2:
                raise ValueError(f"Failed to extract 2D array, got shape {tensor_np.shape}")
                
            return tensor_np
        
        # Process tensors with the helper function
        try:
            pred_np = extract_2d_slice(pred_dmap, sample_idx)
            snake_np = extract_2d_slice(snake_dm, sample_idx)
            print(f"Successfully extracted 2D slices - pred: {pred_np.shape}, snake: {snake_np.shape}")
        except Exception as e:
            print(f"Error extracting 2D slices: {str(e)}")
            raise
        
        # Verify shapes are correct
        print(f"DEBUG: Final shapes - pred_np: {pred_np.shape}, snake_np: {snake_np.shape}")
        
        # Get min and max values for consistent colormaps
        vmin = min(pred_np.min(), snake_np.min())
        vmax = max(pred_np.max(), snake_np.max())
        
        # Create a custom diverging colormap with white at zero
        # Blue for negative (vessels), Red for positive (background)
        colors = [(0, 0, 1), (0.8, 0.8, 1), (1, 1, 1), (1, 0.8, 0.8), (1, 0, 0)]
        positions = [0, 0.4, 0.5, 0.6, 1]
        cmap = LinearSegmentedColormap.from_list("vessel_map", list(zip(positions, colors)))
        
        # Create figure with 3 rows and 2 columns
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        fig.suptitle(f"Snake Distance Map Visualization - Epoch {epoch}", fontsize=16)
        
        # Row 1: Distance maps with continuous colors
        im1 = axes[0, 0].imshow(pred_np, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f'Prediction Distance Map\nMin: {pred_np.min():.2f}, Max: {pred_np.max():.2f}')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(snake_np, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f'Snake Distance Map (Ground Truth)\nMin: {snake_np.min():.2f}, Max: {snake_np.max():.2f}')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Row 2: Binary vessel segmentations (negative values = vessels)
        pred_binary = (pred_np < 0).astype(np.float32)
        snake_binary = (snake_np < 0).astype(np.float32)
        
        vessel_cmap = plt.cm.Blues
        
        im3 = axes[1, 0].imshow(pred_binary, cmap=vessel_cmap, vmin=0, vmax=1)
        axes[1, 0].set_title(f'Predicted Vessels (Negative Values)\nPixels: {np.sum(pred_binary):.0f} ({100*np.mean(pred_binary):.2f}%)')
        
        im4 = axes[1, 1].imshow(snake_binary, cmap=vessel_cmap, vmin=0, vmax=1)
        axes[1, 1].set_title(f'Ground Truth Vessels (Negative Values)\nPixels: {np.sum(snake_binary):.0f} ({100*np.mean(snake_binary):.2f}%)')
        
        # Row 3: Difference between maps and distribution histograms
        diff = pred_np - snake_np
        im5 = axes[2, 0].imshow(diff, cmap='RdBu_r')
        axes[2, 0].set_title(f'Difference (Prediction - Ground Truth)\nMin: {diff.min():.2f}, Max: {diff.max():.2f}')
        plt.colorbar(im5, ax=axes[2, 0])
        
        # Histograms of values
        axes[2, 1].hist(snake_np.flatten(), bins=50, alpha=0.5, color='blue', label='Ground Truth')
        axes[2, 1].hist(pred_np.flatten(), bins=50, alpha=0.5, color='red', label='Prediction')
        axes[2, 1].set_title('Value Distribution')
        axes[2, 1].axvline(x=0, color='black', linestyle='--', label='Zero')
        axes[2, 1].legend()
        
        # Set axes off for all subplots
        for i in range(3):
            for j in range(2):
                if i == 2 and j == 1:  # Skip histogram
                    continue
                axes[i, j].axis('off')
        
        # Save the figure
        filename = f"snake_viz_epoch_{epoch}_sample_{sample_idx}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, filename), dpi=150)
        plt.close()
        print(f"Saved visualization to {os.path.join(self.vis_dir, filename)}")
        
        return fig
    
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
    
    
