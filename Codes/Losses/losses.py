
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
                 cropsz, dmax, maxedgelen, extgradfac, slow_start, negative_weight=0.5,
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
        self.enhancement_factor = 4.0
        
        self.visualize_maps = True
        self.vis_dir = 'snake_visualizations'
        self.vis_seed = vis_seed
        self.vis_sample_index = vis_sample_index
        
        random.seed(self.vis_seed)
        np.random.seed(self.vis_seed)
        torch.manual_seed(self.vis_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.vis_seed)

        self.printed = -1

    def cuda(self):
        super(SnakeFastLoss, self).cuda()
        self.fltrt = self.fltrt.cuda()
        self.iscuda = True
        return self

    def forward(self, pred_dmap, lbl_graphs, crops=None, epoch=float("inf")):
        # pred_dmap is the predicted distance map from the UNet
        # lbl_graphs contains graphs each represent a label as a snake
        # crops is a list of slices, each represents the crop area of the corresponding snake

        pred_ = pred_dmap
        gimgW = cmptGradIm(pred_.abs(), self.fltrt)
        gimg = cmptGradIm(pred_, self.fltrt)
        gimg *= self.extgradfac
        gimgW *= self.extgradfac
        snake_dmap = []
        optimized_graphs = [None] * len(lbl_graphs)

        for i, lg in enumerate(zip(lbl_graphs, gimg, gimgW)):
            # i is index num
            # lg is a tuple of a graph and a gradient image
            l = lg[0]  # graph
            g = lg[1]  # gradient image
            gw = lg[2]

            if crops:
                crop = crops[i]
            else:
                crop=[slice(0,s) for s in g.shape[1:]]

            s = GradImRib(graph=l, crop=crop, stepsz=self.stepsz, alpha=self.alpha,
                        beta=self.beta,dim=self.ndims, gimgV=g, gimgW=gw)

            if self.iscuda: 
                s.cuda()
            if self.slow_start < epoch:
                s.optim(self.nsteps, self.nsteps_width)

            optimized_graphs[i] = s.getGraph()

            dmap = s.render_distance_map_with_widths_cropped(g[0].shape, self.cropsz, self.dmax, self.maxedgelen)
            # make everywhere outside of the snake self.dmax
            #dmap[dmap>0] = self.dmax
            snake_dmap.append(dmap)

        snake_dm = torch.stack(snake_dmap, 0).unsqueeze(1) 

        snake_min, snake_max = snake_dm.min().item(), snake_dm.max().item()
        snake_negative_mask = snake_dm < 0
        snake_dm[snake_negative_mask] *= self.enhancement_factor
            
        self.snake_dm = snake_dm
            
        if epoch % 10 == 0 and self.printed != epoch:
            self.printed = epoch
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
                    graph_to_visualize = None
                    if self.vis_sample_index < len(optimized_graphs):
                        graph_to_visualize = optimized_graphs[self.vis_sample_index]
                    else:
                        print(f"Warning: vis_sample_index {self.vis_sample_index} is out of bounds for optimized_graphs (len {len(optimized_graphs)}). Visualizing with no graph overlay.")
                    
                    self.visualize_distance_maps(pred_dmap, snake_dm, graph_to_visualize, epoch, sample_idx=self.vis_sample_index)
                except Exception as e:
                    print(f"Warning: Could not create visualization: {str(e)}")
        
        squared_diff = (pred_dmap - snake_dm)**2
        
        negative_mask = (snake_dm < 0).float()
        weighted_squared_diff = squared_diff * (1.0 + negative_mask * self.negative_weight)
        loss = weighted_squared_diff.mean()

        """ if epoch % 50 == 0:
            print(f"Loss: {loss.item():.4f}")
            print("=" * 50) """
        
        self.snake = s
        return loss

    def visualize_distance_maps(self, pred_dmap, snake_dm, current_graph, epoch, sample_idx=None):
        """
        Visualize the predicted distance map and snake target map.
        For 3D, shows X, Y, Z minimum intensity projections.
        For 2D, shows the 2D slice.
        
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
        
        if sample_idx is None:
            sample_idx = self.vis_sample_index
            
        print(f"Visualizing: epoch={epoch}, sample_idx={sample_idx}, ndims={self.ndims}, seed={self.vis_seed}")
        
        print(f"Initial pred_dmap shape: {pred_dmap.shape}, snake_dm shape: {snake_dm.shape}")

        fig = None

        if self.ndims == 3:
            def get_3d_numpy_array(tensor, s_idx):
                tensor_np = tensor.detach().cpu().numpy()
                if tensor_np.ndim == 5: # B, C, D, H, W
                    tensor_np = tensor_np[s_idx, 0, :, :, :] # Select sample and first channel
                elif tensor_np.ndim == 4: # B, D, H, W or C, D, H, W
                    if tensor_np.shape[0] == pred_dmap.shape[0] and tensor_np.shape[0] > 1: # Batch dim present
                         tensor_np = tensor_np[s_idx, :, :, :]
                    else: # Channel dim present (or batch size 1)
                         tensor_np = tensor_np[0, :, :, :] 
                elif tensor_np.ndim == 3: # D, H, W
                    pass # Already in correct shape
                else:
                    raise ValueError(f"Unsupported tensor shape for 3D visualization: {tensor_np.shape}")
                
                if tensor_np.ndim != 3:
                     raise ValueError(f"Failed to extract 3D array, got shape {tensor_np.shape}")
                return tensor_np

            try:
                pred_np_3d = get_3d_numpy_array(pred_dmap, sample_idx)
                snake_np_3d = get_3d_numpy_array(snake_dm, sample_idx)
                print(f"Processed 3D shapes - pred: {pred_np_3d.shape}, snake: {snake_np_3d.shape}")
            except Exception as e:
                print(f"Error extracting 3D volumes: {str(e)}")
                return None

            fig, axes = plt.subplots(6, 3, figsize=(18, 28)) # 5 rows for slices, 1 for histogram
            fig.suptitle(f"3D Snake Distance Map Visualization - Epoch {epoch}, Sample {sample_idx}", fontsize=16)

            vmin = min(pred_np_3d.min(), snake_np_3d.min())
            vmax = max(pred_np_3d.max(), snake_np_3d.max())
            continuous_cmap_colors = [(0, 0, 1), (0.8, 0.8, 1), (1, 1, 1), (1, 0.8, 0.8), (1, 0, 0)]
            continuous_cmap_positions = [0, 0.4, 0.5, 0.6, 1]
            continuous_cmap = LinearSegmentedColormap.from_list("vessel_map_3d", list(zip(continuous_cmap_positions, continuous_cmap_colors)))
            binary_cmap = plt.cm.Blues
            diff_cmap = 'RdBu_r'

            datasets = [
                (pred_np_3d, f'Prediction DM\nMin: {pred_np_3d.min():.2f}, Max: {pred_np_3d.max():.2f}', continuous_cmap, {'vmin': vmin, 'vmax': vmax}),
                (snake_np_3d, f'Snake DM (GT)\nMin: {snake_np_3d.min():.2f}, Max: {snake_np_3d.max():.2f}', continuous_cmap, {'vmin': vmin, 'vmax': vmax})
            ]

            for i, (data_3d, title_prefix, cmap, cmap_args) in enumerate(datasets):
                projections = [
                    (np.min(data_3d, axis=2), "(Y-Z Projection / Min X)"),
                    (np.min(data_3d, axis=1), "(X-Z Projection / Min Y)"),
                    (np.min(data_3d, axis=0), "(X-Y Projection / Min Z)") 
                ]
                for j, (slice_data, proj_title) in enumerate(projections):
                    im = axes[i, j].imshow(slice_data.T, cmap=cmap, origin='lower', **cmap_args)
                    axes[i, j].set_title(f"{title_prefix}\n{proj_title}")
                    fig.colorbar(im, ax=axes[i, j], orientation='horizontal', fraction=0.046, pad=0.1)
                    axes[i, j].axis('off')

                    # Plot graph overlay
                    if current_graph:
                        for node1_idx, node2_idx in current_graph.edges:
                            if node1_idx in current_graph.nodes and node2_idx in current_graph.nodes:
                                pos1 = current_graph.nodes[node1_idx]['pos']
                                pos2 = current_graph.nodes[node2_idx]['pos']
                                if j == 0: # Y-Z plane, imshow x-axis=Z, y-axis=Y
                                    axes[i, j].plot([pos1[2], pos2[2]], [pos1[1], pos2[1]], 'r-', linewidth=1.0)
                                elif j == 1: # X-Z plane, imshow x-axis=Z, y-axis=X
                                    axes[i, j].plot([pos1[2], pos2[2]], [pos1[0], pos2[0]], 'r-', linewidth=1.0)
                                elif j == 2: # X-Y plane, imshow x-axis=Y, y-axis=X
                                    axes[i, j].plot([pos1[1], pos2[1]], [pos1[0], pos2[0]], 'r-', linewidth=1.0)
            
            pred_binary_3d = (pred_np_3d < 0).astype(np.float32)
            snake_binary_3d = (snake_np_3d < 0).astype(np.float32)
            binary_data_list = [
                (pred_binary_3d, f'Predicted Vessels (Negative)\nPixels: {np.sum(pred_binary_3d):.0f} ({100*np.mean(pred_binary_3d):.2f}%)'),
                (snake_binary_3d, f'GT Vessels (Negative)\nPixels: {np.sum(snake_binary_3d):.0f} ({100*np.mean(snake_binary_3d):.2f}%)')
            ]
            plot_row = 2 # Initialize plot_row for binary images
            for data_3d_bin, title_prefix_bin in binary_data_list:
                # Assuming D,H,W maps to Z,Y,X for data_3d_bin as well
                projections_bin = [
                    (np.max(data_3d_bin, axis=2), "(Y-Z Max Proj)"), # Project along X-axis
                    (np.max(data_3d_bin, axis=1), "(X-Z Max Proj)"), # Project along Y-axis
                    (np.max(data_3d_bin, axis=0), "(X-Y Max Proj)")  # Project along Z-axis
                ]
                for j, (slice_data, proj_title) in enumerate(projections_bin):
                    axes[plot_row, j].imshow(slice_data.T, cmap=binary_cmap, origin='lower', vmin=0, vmax=1)
            
            # Difference Plot
            diff_3d = pred_np_3d - snake_np_3d
            diff_projections = [
                (np.min(diff_3d, axis=2), "(Y-Z Projection / Min X)"), # Project along X-axis
                (np.min(diff_3d, axis=1), "(X-Z Projection / Min Y)"), # Project along Y-axis
                (np.min(diff_3d, axis=0), "(X-Y Projection / Min Z)")  # Project along Z-axis
            ]
            diff_title_prefix = f'Difference (Pred - GT)\nMin: {diff_3d.min():.2f}, Max: {diff_3d.max():.2f}'
            for j, (slice_data, proj_title) in enumerate(diff_projections):
                im = axes[4, j].imshow(slice_data.T, cmap=diff_cmap, origin='lower')
                axes[4, j].set_title(f"{diff_title_prefix}\n{proj_title}")
                fig.colorbar(im, ax=axes[4, j], orientation='horizontal', fraction=0.046, pad=0.1)
                axes[4, j].axis('off')

            axes[5, 0].hist(snake_np_3d.flatten(), bins=50, alpha=0.5, color='blue', label='Ground Truth DM')
            axes[5, 0].hist(pred_np_3d.flatten(), bins=50, alpha=0.5, color='red', label='Prediction DM')
            axes[5, 0].set_title('Distance Map Value Distribution')
            axes[5, 0].axvline(x=0, color='black', linestyle='--', label='Zero')
            axes[5, 0].legend()
            axes[5, 0].set_xlabel("Distance Value")
            axes[5, 0].set_ylabel("Frequency")
            axes[5, 1].axis('off')
            axes[5, 2].axis('off')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            filename = f"snake_viz_epoch_{epoch}_sample_{sample_idx}_3D.png"
            plt.savefig(os.path.join(self.vis_dir, filename), dpi=150)
            plt.close(fig)
            print(f"Saved 3D visualization to {os.path.join(self.vis_dir, filename)}")

        elif self.ndims == 2:
            def extract_2d_slice(tensor, idx=0): # Original helper for 2D
                tensor_np = tensor.detach().cpu().numpy()
                while tensor_np.ndim > 2:
                    if tensor_np.shape[0] > 1:
                        if idx < tensor_np.shape[0]:
                            tensor_np = tensor_np[idx]
                        else:
                            print(f"Warning: Index {idx} out of bounds for dim size {tensor_np.shape[0]}. Using 0.")
                            tensor_np = tensor_np[0]
                    else:
                        tensor_np = tensor_np.squeeze(0)
                    if tensor_np.ndim <= 2: break
                tensor_np = np.squeeze(tensor_np)
                if tensor_np.ndim != 2:
                    raise ValueError(f"Failed to extract 2D array, got shape {tensor_np.shape}")
                return tensor_np
            
            try:
                pred_np = extract_2d_slice(pred_dmap, sample_idx)
                snake_np = extract_2d_slice(snake_dm, sample_idx)
                print(f"Processed 2D shapes - pred: {pred_np.shape}, snake: {snake_np.shape}")
            except Exception as e:
                print(f"Error extracting 2D slices: {str(e)}")
                return None

            fig, axes = plt.subplots(3, 2, figsize=(14, 14))
            fig.suptitle(f"2D Snake Distance Map Visualization - Epoch {epoch}, Sample {sample_idx}", fontsize=16)

            vmin = min(pred_np.min(), snake_np.min())
            vmax = max(pred_np.max(), snake_np.max())
            colors = [(0, 0, 1), (0.8, 0.8, 1), (1, 1, 1), (1, 0.8, 0.8), (1, 0, 0)]
            positions = [0, 0.4, 0.5, 0.6, 1]
            cmap = LinearSegmentedColormap.from_list("vessel_map_2d", list(zip(positions, colors)))
            
            im1 = axes[0, 0].imshow(pred_np, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            axes[0, 0].set_title(f'Prediction DM\nMin: {pred_np.min():.2f}, Max: {pred_np.max():.2f}')
            plt.colorbar(im1, ax=axes[0, 0])
            if current_graph:
                for node1_idx, node2_idx in current_graph.edges:
                    if node1_idx in current_graph.nodes and node2_idx in current_graph.nodes:
                        pos1 = current_graph.nodes[node1_idx]['pos']
                        pos2 = current_graph.nodes[node2_idx]['pos']
                        axes[0, 0].plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'r-', linewidth=1.0)

            im2 = axes[0, 1].imshow(snake_np, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            axes[0, 1].set_title(f'Snake DM (GT)\nMin: {snake_np.min():.2f}, Max: {snake_np.max():.2f}')
            plt.colorbar(im2, ax=axes[0, 1])
            if current_graph:
                for node1_idx, node2_idx in current_graph.edges:
                    if node1_idx in current_graph.nodes and node2_idx in current_graph.nodes:
                        pos1 = current_graph.nodes[node1_idx]['pos']
                        pos2 = current_graph.nodes[node2_idx]['pos']
                        axes[0, 1].plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'r-', linewidth=1.0)
            
            pred_binary = (pred_np < 0).astype(np.float32)
            snake_binary = (snake_np < 0).astype(np.float32)
            vessel_cmap = plt.cm.Blues
            
            axes[1, 0].imshow(pred_binary, cmap=vessel_cmap, vmin=0, vmax=1, origin='lower')
            axes[1, 0].set_title(f'Predicted Vessels (Negative)\nPixels: {np.sum(pred_binary):.0f} ({100*np.mean(pred_binary):.2f}%)')
            
            axes[1, 1].imshow(snake_binary, cmap=vessel_cmap, vmin=0, vmax=1, origin='lower')
            axes[1, 1].set_title(f'GT Vessels (Negative)\nPixels: {np.sum(snake_binary):.0f} ({100*np.mean(snake_binary):.2f}%)')
            
            diff = pred_np - snake_np
            im5 = axes[2, 0].imshow(diff, cmap='RdBu_r', origin='lower')
            axes[2, 0].set_title(f'Difference (Pred - GT)\nMin: {diff.min():.2f}, Max: {diff.max():.2f}')
            plt.colorbar(im5, ax=axes[2, 0])
            
            axes[2, 1].hist(snake_np.flatten(), bins=50, alpha=0.5, color='blue', label='Ground Truth DM')
            axes[2, 1].hist(pred_np.flatten(), bins=50, alpha=0.5, color='red', label='Prediction DM')
            axes[2, 1].set_title('Value Distribution')
            axes[2, 1].axvline(x=0, color='black', linestyle='--', label='Zero')
            axes[2, 1].legend()
            
            for i_ax in range(3):
                for j_ax in range(2):
                    if not (i_ax == 2 and j_ax == 1): # Skip histogram
                        axes[i_ax, j_ax].axis('off')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            filename = f"snake_viz_epoch_{epoch}_sample_{sample_idx}_2D.png"
            plt.savefig(os.path.join(self.vis_dir, filename), dpi=150)
            plt.close(fig)
            print(f"Saved 2D visualization to {os.path.join(self.vis_dir, filename)}")
        else:
            print(f"Visualization for ndims={self.ndims} is not implemented.")
            return None
            
        return fig
    
class SnakeSimpleLoss(nn.Module):
    def __init__(self, stepsz, alpha, beta, fltrstdev, ndims, nsteps, nsteps_width,
                 cropsz, dmax, maxedgelen, extgradfac, slow_start, negative_weight=1.5,
                 vis_seed=42, vis_sample_index=0):
        super(SnakeSimpleLoss, self).__init__()
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
        self.enhancement_factor = 2.0
        
        self.visualize_maps = True
        self.vis_dir = 'snake_visualizations'
        self.vis_seed = vis_seed
        self.vis_sample_index = vis_sample_index
        
        random.seed(self.vis_seed)
        np.random.seed(self.vis_seed)
        torch.manual_seed(self.vis_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.vis_seed)

    def cuda(self):
        super(SnakeSimpleLoss, self).cuda()
        self.fltrt = self.fltrt.cuda()
        self.iscuda = True
        return self

    def forward(self, pred_dmap, lbl_graphs, crops=None, epoch=float("inf")):
        # pred_dmap is the predicted distance map from the UNet
        # lbl_graphs contains graphs each represent a label as a snake
        # crops is a list of slices, each represents the crop area of the corresponding snake

        pred_ = pred_dmap
        gimgW = cmptGradIm(pred_.abs(), self.fltrt)
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

            if crops:
                crop = crops[i]
            else:
                crop=[slice(0,s) for s in g.shape[1:]]

            s = GradImRib(graph=l, crop=crop, stepsz=self.stepsz, alpha=self.alpha,
                        beta=self.beta,dim=self.ndims, gimgV=g, gimgW=gw)
                    
            if self.iscuda: 
                s.cuda()
            if self.slow_start < epoch:
                s.optim(self.nsteps, self.nsteps_width)

            dmap = s.render_distance_map_with_widths_cropped(g[1:].shape, self.cropsz, self.dmax, self.maxedgelen)
            # make everywhere outside of the snake self.dmax
            dmap[dmap>0] = self.dmax
            snake_dmap.append(dmap)

        snake_dm = torch.stack(snake_dmap, 0).unsqueeze(1) 

        snake_min, snake_max = snake_dm.min().item(), snake_dm.max().item()
        snake_negative_mask = snake_dm < 0
        snake_dm[snake_negative_mask] *= self.enhancement_factor
            
        self.snake_dm = snake_dm
            
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
        
        squared_diff = (pred_dmap - snake_dm.detach())**2
        
        negative_mask = (snake_dm < 0).float()
        weighted_squared_diff = squared_diff * (1.0 + negative_mask * self.negative_weight)
        loss = weighted_squared_diff.mean()

        if epoch % 50 == 0:
            print(f"Loss: {loss.item():.4f}")
            print("=" * 50)
        
        self.snake = s
        return loss
    
    def visualize_distance_maps(self, pred_dmap, snake_dm, epoch, sample_idx=None):
        """
        Visualize the predicted distance map and snake target map.
        For 3D, shows X, Y, Z minimum intensity projections.
        For 2D, shows the 2D slice.
        
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
        
        if sample_idx is None:
            sample_idx = self.vis_sample_index
            
        print(f"Visualizing: epoch={epoch}, sample_idx={sample_idx}, ndims={self.ndims}, seed={self.vis_seed}")
        
        print(f"Initial pred_dmap shape: {pred_dmap.shape}, snake_dm shape: {snake_dm.shape}")

        fig = None

        if self.ndims == 3:
            def get_3d_numpy_array(tensor, s_idx):
                tensor_np = tensor.detach().cpu().numpy()
                if tensor_np.ndim == 5: # B, C, D, H, W
                    tensor_np = tensor_np[s_idx, 0, :, :, :] # Select sample and first channel
                elif tensor_np.ndim == 4: # B, D, H, W or C, D, H, W
                    if tensor_np.shape[0] == pred_dmap.shape[0] and tensor_np.shape[0] > 1: # Batch dim present
                         tensor_np = tensor_np[s_idx, :, :, :]
                    else: # Channel dim present (or batch size 1)
                         tensor_np = tensor_np[0, :, :, :] 
                elif tensor_np.ndim == 3: # D, H, W
                    pass # Already in correct shape
                else:
                    raise ValueError(f"Unsupported tensor shape for 3D visualization: {tensor_np.shape}")
                
                if tensor_np.ndim != 3:
                     raise ValueError(f"Failed to extract 3D array, got shape {tensor_np.shape}")
                return tensor_np

            try:
                pred_np_3d = get_3d_numpy_array(pred_dmap, sample_idx)
                snake_np_3d = get_3d_numpy_array(snake_dm, sample_idx)
                print(f"Processed 3D shapes - pred: {pred_np_3d.shape}, snake: {snake_np_3d.shape}")
            except Exception as e:
                print(f"Error extracting 3D volumes: {str(e)}")
                return None

            fig, axes = plt.subplots(6, 3, figsize=(18, 28)) # 5 rows for slices, 1 for histogram
            fig.suptitle(f"3D Snake Distance Map Visualization - Epoch {epoch}, Sample {sample_idx}", fontsize=16)

            vmin = min(pred_np_3d.min(), snake_np_3d.min())
            vmax = max(pred_np_3d.max(), snake_np_3d.max())
            continuous_cmap_colors = [(0, 0, 1), (0.8, 0.8, 1), (1, 1, 1), (1, 0.8, 0.8), (1, 0, 0)]
            continuous_cmap_positions = [0, 0.4, 0.5, 0.6, 1]
            continuous_cmap = LinearSegmentedColormap.from_list("vessel_map_3d", list(zip(continuous_cmap_positions, continuous_cmap_colors)))
            binary_cmap = plt.cm.Blues
            diff_cmap = 'RdBu_r'

            datasets = [
                (pred_np_3d, f'Prediction DM\nMin: {pred_np_3d.min():.2f}, Max: {pred_np_3d.max():.2f}', continuous_cmap, {'vmin': vmin, 'vmax': vmax}),
                (snake_np_3d, f'Snake DM (GT)\nMin: {snake_np_3d.min():.2f}, Max: {snake_np_3d.max():.2f}', continuous_cmap, {'vmin': vmin, 'vmax': vmax})
            ]

            for i, (data_3d, title_prefix, cmap, cmap_args) in enumerate(datasets):
                projections = [
                    (np.min(data_3d, axis=0), "(Y-Z Projection / Min X)"),
                    (np.min(data_3d, axis=1), "(X-Z Projection / Min Y)"),
                    (np.min(data_3d, axis=2), "(X-Y Projection / Min Z)")
                ]
                for j, (slice_data, proj_title) in enumerate(projections):
                    im = axes[i, j].imshow(slice_data.T, cmap=cmap, origin='lower', **cmap_args)
                    axes[i, j].set_title(f"{title_prefix}\n{proj_title}")
                    fig.colorbar(im, ax=axes[i, j], orientation='horizontal', fraction=0.046, pad=0.1)
                    axes[i, j].axis('off')
            
            pred_binary_3d = (pred_np_3d < 0).astype(np.float32)
            snake_binary_3d = (snake_np_3d < 0).astype(np.float32)
            binary_datasets = [
                (pred_binary_3d, f'Predicted Vessels (Negative)\nPixels: {np.sum(pred_binary_3d):.0f} ({100*np.mean(pred_binary_3d):.2f}%)', binary_cmap, {'vmin':0, 'vmax':1}),
                (snake_binary_3d, f'GT Vessels (Negative)\nPixels: {np.sum(snake_binary_3d):.0f} ({100*np.mean(snake_binary_3d):.2f}%)', binary_cmap, {'vmin':0, 'vmax':1})
            ]

            for i, (data_3d, title_prefix, cmap, cmap_args) in enumerate(binary_datasets):
                current_row = i + 2 
                projections = [
                    (np.min(data_3d, axis=0), "(Y-Z Projection / Min X)"),
                    (np.min(data_3d, axis=1), "(X-Z Projection / Min Y)"),
                    (np.min(data_3d, axis=2), "(X-Y Projection / Min Z)")
                ]
                for j, (slice_data, proj_title) in enumerate(projections):
                    axes[current_row, j].imshow(slice_data.T, cmap=cmap, origin='lower', **cmap_args)
                    axes[current_row, j].set_title(f"{title_prefix}\n{proj_title}")
                    axes[current_row, j].axis('off')

            diff_3d = pred_np_3d - snake_np_3d
            diff_projections = [
                (np.min(diff_3d, axis=0), "(Y-Z Projection / Min X)"),
                (np.min(diff_3d, axis=1), "(X-Z Projection / Min Y)"),
                (np.min(diff_3d, axis=2), "(X-Y Projection / Min Z)")
            ]
            diff_title_prefix = f'Difference (Pred - GT)\nMin: {diff_3d.min():.2f}, Max: {diff_3d.max():.2f}'
            for j, (slice_data, proj_title) in enumerate(diff_projections):
                im = axes[4, j].imshow(slice_data.T, cmap=diff_cmap, origin='lower')
                axes[4, j].set_title(f"{diff_title_prefix}\n{proj_title}")
                fig.colorbar(im, ax=axes[4, j], orientation='horizontal', fraction=0.046, pad=0.1)
                axes[4, j].axis('off')

            axes[5, 0].hist(snake_np_3d.flatten(), bins=50, alpha=0.5, color='blue', label='Ground Truth DM')
            axes[5, 0].hist(pred_np_3d.flatten(), bins=50, alpha=0.5, color='red', label='Prediction DM')
            axes[5, 0].set_title('Distance Map Value Distribution')
            axes[5, 0].axvline(x=0, color='black', linestyle='--', label='Zero')
            axes[5, 0].legend()
            axes[5, 0].set_xlabel("Distance Value")
            axes[5, 0].set_ylabel("Frequency")
            axes[5, 1].axis('off')
            axes[5, 2].axis('off')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            filename = f"snake_viz_epoch_{epoch}_sample_{sample_idx}_3D.png"
            plt.savefig(os.path.join(self.vis_dir, filename), dpi=150)
            plt.close(fig)
            print(f"Saved 3D visualization to {os.path.join(self.vis_dir, filename)}")

        elif self.ndims == 2:
            def extract_2d_slice(tensor, idx=0): # Original helper for 2D
                tensor_np = tensor.detach().cpu().numpy()
                while tensor_np.ndim > 2:
                    if tensor_np.shape[0] > 1:
                        if idx < tensor_np.shape[0]:
                            tensor_np = tensor_np[idx]
                        else:
                            print(f"Warning: Index {idx} out of bounds for dim size {tensor_np.shape[0]}. Using 0.")
                            tensor_np = tensor_np[0]
                    else:
                        tensor_np = tensor_np.squeeze(0)
                    if tensor_np.ndim <= 2: break
                tensor_np = np.squeeze(tensor_np)
                if tensor_np.ndim != 2:
                    raise ValueError(f"Failed to extract 2D array, got shape {tensor_np.shape}")
                return tensor_np
            
            try:
                pred_np = extract_2d_slice(pred_dmap, sample_idx)
                snake_np = extract_2d_slice(snake_dm, sample_idx)
                print(f"Processed 2D shapes - pred: {pred_np.shape}, snake: {snake_np.shape}")
            except Exception as e:
                print(f"Error extracting 2D slices: {str(e)}")
                return None

            fig, axes = plt.subplots(3, 2, figsize=(14, 14))
            fig.suptitle(f"2D Snake Distance Map Visualization - Epoch {epoch}, Sample {sample_idx}", fontsize=16)

            vmin = min(pred_np.min(), snake_np.min())
            vmax = max(pred_np.max(), snake_np.max())
            colors = [(0, 0, 1), (0.8, 0.8, 1), (1, 1, 1), (1, 0.8, 0.8), (1, 0, 0)]
            positions = [0, 0.4, 0.5, 0.6, 1]
            cmap = LinearSegmentedColormap.from_list("vessel_map_2d", list(zip(positions, colors)))
            
            im1 = axes[0, 0].imshow(pred_np, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            axes[0, 0].set_title(f'Prediction DM\nMin: {pred_np.min():.2f}, Max: {pred_np.max():.2f}')
            plt.colorbar(im1, ax=axes[0, 0])
            
            im2 = axes[0, 1].imshow(snake_np, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            axes[0, 1].set_title(f'Snake DM (GT)\nMin: {snake_np.min():.2f}, Max: {snake_np.max():.2f}')
            plt.colorbar(im2, ax=axes[0, 1])
            
            pred_binary = (pred_np < 0).astype(np.float32)
            snake_binary = (snake_np < 0).astype(np.float32)
            vessel_cmap = plt.cm.Blues
            
            axes[1, 0].imshow(pred_binary, cmap=vessel_cmap, vmin=0, vmax=1, origin='lower')
            axes[1, 0].set_title(f'Predicted Vessels (Negative)\nPixels: {np.sum(pred_binary):.0f} ({100*np.mean(pred_binary):.2f}%)')
            
            axes[1, 1].imshow(snake_binary, cmap=vessel_cmap, vmin=0, vmax=1, origin='lower')
            axes[1, 1].set_title(f'GT Vessels (Negative)\nPixels: {np.sum(snake_binary):.0f} ({100*np.mean(snake_binary):.2f}%)')
            
            diff = pred_np - snake_np
            im5 = axes[2, 0].imshow(diff, cmap='RdBu_r', origin='lower')
            axes[2, 0].set_title(f'Difference (Pred - GT)\nMin: {diff.min():.2f}, Max: {diff.max():.2f}')
            plt.colorbar(im5, ax=axes[2, 0])
            
            axes[2, 1].hist(snake_np.flatten(), bins=50, alpha=0.5, color='blue', label='Ground Truth DM')
            axes[2, 1].hist(pred_np.flatten(), bins=50, alpha=0.5, color='red', label='Prediction DM')
            axes[2, 1].set_title('Value Distribution')
            axes[2, 1].axvline(x=0, color='black', linestyle='--', label='Zero')
            axes[2, 1].legend()
            
            for i_ax in range(3):
                for j_ax in range(2):
                    if not (i_ax == 2 and j_ax == 1): # Skip histogram
                        axes[i_ax, j_ax].axis('off')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            filename = f"snake_viz_epoch_{epoch}_sample_{sample_idx}_2D.png"
            plt.savefig(os.path.join(self.vis_dir, filename), dpi=150)
            plt.close(fig)
            print(f"Saved 2D visualization to {os.path.join(self.vis_dir, filename)}")
        else:
            print(f"Visualization for ndims={self.ndims} is not implemented.")
            return None
            
        return fig
    