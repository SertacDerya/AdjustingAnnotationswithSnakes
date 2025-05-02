from .snake import Snake
import torch
import torch.nn.functional as F
import math

def cmptExtGrad(snakepos,eGradIm):
    # returns the values of eGradIm at positions snakepos
    # snakepos  is a k X d matrix, where snakepos[j,:] represents a d-dimensional position of the j-th node of the snake
    # eGradIm   is a tensor containing the energy gradient image, either of size
    #           3 X d X h X w, for 3D, or of size
    #           2     X h X w, for 2D snakes
    # returns a tensor of the same size as snakepos,
    # containing the values of eGradIm at coordinates specified by snakepos
    
    # scale snake coordinates to match the hilarious requirements of grid_sample
    # we use the obsolote convention, where align_corners=True
    scale=torch.tensor(eGradIm.shape[1:]).reshape((1,-1)).type_as(snakepos)-1.0
    sp=2*snakepos/scale-1.0
    
    if eGradIm.shape[0]==3:
        # invert the coordinate order to match other hilarious specs of grid_sample
        spi=torch.einsum('km,md->kd',[sp,torch.tensor([[0,0,1],[0,1,0],[1,0,0]]).type_as(sp).to(sp.device)])
        egrad=torch.nn.functional.grid_sample(eGradIm[None],spi[None,None,None],
                                           align_corners=True)
        egrad=egrad.permute(0,2,3,4,1)
    if eGradIm.shape[0]==2:
        # invert the coordinate order to match other hilarious specs of grid_sample
        spi=torch.einsum('kl,ld->kd',[sp,torch.tensor([[0,1],[1,0]]).type_as(sp).to(sp.device)])
        egrad=torch.nn.functional.grid_sample(eGradIm[None],spi[None,None],
                                           align_corners=True)
        egrad=egrad.permute(0,2,3,1)
        
    return egrad.reshape_as(snakepos)

class RibbonSnake(Snake):
    def __init__(self, graph, crop, stepsz, alpha, beta, dim):
        # In the new version grad will be separate, so image gradients will not be here
        # Normal initialization of Snake super class
        super().__init__(graph, crop, stepsz, alpha, beta, dim)
        # Additionally we sample from a normal distrubution for widths of nodes
        #self.w = torch.randn(self.s.shape[0]).abs()
        self.w = torch.randint(low=2, high=9, size=(self.s.shape[0],), dtype=torch.float32)

    def cuda(self):
        super().cuda()
        # move the widths to gpu
        self.w = self.w.cuda()

    def set_w(self, widths):
        self.w = widths

    def get_w(self):
        return self.w

    def _compute_normals(self, pos):
        """
        Compute normals (and tangents for 3D) for each center point.
        Returns:
         - 2D: (normals,) where normals is (N,2)
         - 3D: (n1, n2, tangents) each (N,3)
        """
        N, d = pos.shape
        eps = 1e-8
        t = torch.zeros_like(pos)
        if N > 1:
            t[1:-1] = (pos[2:] - pos[:-2]) * 0.5
            t[0] = pos[1] - pos[0]
            t[-1] = pos[-1] - pos[-2]
        t = t / (t.norm(dim=1, keepdim=True) + eps)

        if self.ndims == 2:
            normals = torch.stack([-t[:,1], t[:,0]], dim=1)
            normals = normals / (normals.norm(dim=1, keepdim=True) + eps)
            return (normals,)
        else:
            a = torch.zeros_like(pos)
            a[:] = torch.tensor([1.0, 0.0, 0.0], device=pos.device)
            mask = (t * a).abs().sum(dim=1) > 0.9
            a[mask] = torch.tensor([0.0, 1.0, 0.0], device=pos.device)
            n1 = torch.cross(t, a, dim=1)
            n1 = n1 / (n1.norm(dim=1, keepdim=True) + eps)
            n2 = torch.cross(t, n1, dim=1)
            n2 = n2 / (n2.norm(dim=1, keepdim=True) + eps)
            return (n1, n2, t)
        
    def comp_second_deriv(self):
        """
        Computes an internal smoothness term for widths, related to the second derivative,
        using 1D convolution. Handles boundaries using padding.
        A positive value indicates the width should shrink to increase smoothness,
        a negative value indicates it should grow.
        Returns:
            torch.Tensor: Smoothness term for each node, shape (K, 1).
        """
        K = self.w.numel()
        if K < 3:
             return torch.zeros(K, 1, device=self.w.device, dtype=self.w.dtype)

        w = self.w.view(1, 1, -1)

        kernel = torch.tensor([[[1.0, -2.0, 1.0]]], device=w.device, dtype=w.dtype) # Shape (1, 1, 3)
        padding_size = (kernel.shape[-1] - 1) // 2
        w_padded = F.pad(w, (padding_size, padding_size), mode='replicate')
        smoothness_term = F.conv1d(w_padded, kernel, padding=0)
        return -smoothness_term.view(-1, 1) # Shape (K, 1)

    def step_widths(self, gimgW):
        """
        Update widths by sampling gradient of image W at ribbon edges and
        adding internal smoothness via second derivative.
        Handles leaf nodes using padded convolution for smoothness.
        """
        if self.s.numel() == 0:
            return self.w

        pos = self.s                  # (K, d) tensor of center points
        K, d = pos.shape
        device = pos.device
        w_vec = self.w.view(-1)       # Ensure w is a vector (K,)
        half_r = (w_vec / 2.0).view(K, 1) # Shape (K, 1) for broadcasting

        if d == 2:
            (normals,) = self._compute_normals(pos)
            left_pts  = pos - normals * half_r    
            right_pts = pos + normals * half_r    
            grad_L = cmptExtGrad(left_pts,  gimgW)
            grad_R = cmptExtGrad(right_pts, gimgW)
            grad_w = ((grad_R - grad_L) * normals).sum(dim=1, keepdim=True) # (K, 1)

        elif d == 3:
            n1, n2, _ = self._compute_normals(pos)
            N_samples = 8
            theta = torch.linspace(0, 2*math.pi, N_samples + 1, device=device, dtype=pos.dtype)[:-1]
            theta_exp = theta.view(N_samples, 1, 1)
            n1_exp = n1.unsqueeze(0)                
            n2_exp = n2.unsqueeze(0)                
            half_r_exp = half_r.view(1, K, 1)       
            pos_exp = pos.unsqueeze(0)              
            dirs = theta_exp.cos() * n1_exp + theta_exp.sin() * n2_exp
            epsilon_offset = 1e-4
            pts_out = pos_exp + (half_r_exp + epsilon_offset) * dirs  
            pts_in  = pos_exp - (half_r_exp + epsilon_offset) * dirs  
            all_pts = torch.cat([pts_out, pts_in], dim=0) 
            all_pts_flat = all_pts.view(-1, 3)           
            grads_flat = cmptExtGrad(all_pts_flat, gimgW)
            grads = grads_flat.view(2*N_samples, K, 3)   
            grads_out = grads[:N_samples]
            grads_in  = grads[N_samples:]
            grad_diff = grads_out - grads_in
            radial_component = (grad_diff * dirs).sum(dim=2)
            grad_w = radial_component.mean(dim=0, keepdim=True).t()

        else:
            raise ValueError(f"Unsupported dimension: {d}")

        # --- Internal Smoothness Term ---
        # internal points in the direction width should move to increase smoothness
        internal = self.comp_second_deriv() # (K, 1)

        # --- Combine Terms and Update ---
        # Adaptive alpha balances external gradient and internal smoothness
        # Use max with a small value to prevent division by zero and instability
        alpha = grad_w.abs() / (internal.abs().max(torch.tensor(1e-8, device=device))) # (K, 1)
        # Optional: Clamp alpha to prevent extreme internal forces
        alpha = torch.clamp(alpha, max=10.0)
        total_gradient = -grad_w - alpha * internal

        # gradient descent step: w_new = w_old - stepsz * total_gradient
        self.w = w_vec + self.stepsz * total_gradient.squeeze(-1)
        self.w = torch.clamp(self.w, min=1e-4)
        return self.w
    
    def render_distance_map_with_widths(self, size, max_dist=16.0):
        """
        Unified 2D/3D signed distance map for the ribbon snake using graph structure.

        Args:
            size (tuple): (W, H) for 2D or (X, Y, Z) for 3D grid dimensions.
            max_dist (float): Maximum distance value to clamp to.

        Returns:
            torch.Tensor: Signed distance map of shape `size`. Negative inside,
                          zero on surface, positive outside up to max_dist.
        """
        device = self.s.device
        centers = self.s
        radii = (self.w.flatten() / 2.0)
        eps = 1e-8

        if centers.numel() == 0 or radii.numel() == 0 or len(self.h.nodes) == 0:
            print("Warning: Rendering distance map for empty snake.")
            return torch.full(size, max_dist, device=device, dtype=centers.dtype)

        if centers.shape[0] != radii.shape[0]:
             raise ValueError(f"Mismatch between center points ({centers.shape[0]}) and radii ({radii.shape[0]})")

        axes = [torch.arange(sz, device=device, dtype=torch.float32) for sz in size]
        mesh = torch.meshgrid(*axes, indexing='ij')
        del axes
        points = torch.stack([m.flatten() for m in mesh], dim=1)
        del mesh
        num_points = points.shape[0]
        min_dist = torch.full((num_points,), float('inf'), device=device, dtype=centers.dtype)
        del num_points

        if len(self.h.edges) > 0:
            try:
                if hasattr(self, 'n2i') and self.n2i:
                     edge_indices_list = [(self.n2i[u], self.n2i[v]) for u, v in self.h.edges]
                else:
                     edge_indices_list = list(self.h.edges)

                edge_indices = torch.tensor(edge_indices_list, device=device, dtype=torch.long) # (E, 2)
                del edge_indices_list
            except KeyError as e:
                 raise RuntimeError(f"Node ID {e} from graph edges not found in n2i mapping. Ensure Snake init populated n2i correctly.") from e
            except Exception as e:
                 raise RuntimeError(f"Error processing graph edges. Ensure self.h and self.n2i are correct. Original error: {e}")


            starts = centers[edge_indices[:, 0]]
            ends   = centers[edge_indices[:, 1]]
            r0     = radii[edge_indices[:, 0]]
            r1     = radii[edge_indices[:, 1]]
            del edge_indices

            vec = ends - starts
            L_sq = (vec**2).sum(dim=1)
            valid_edge = L_sq > eps**2
            if torch.any(valid_edge):
                 starts_v, ends_v = starts[valid_edge], ends[valid_edge]
                 r0_v, r1_v = r0[valid_edge], r1[valid_edge]
                 vec_v = vec[valid_edge]
                 L_sq_v = L_sq[valid_edge]
                 L_v = torch.sqrt(L_sq_v)
                 del L_sq_v
                 D_v = vec_v / (L_v.unsqueeze(1) + eps)
                 del vec_v

                 P_exp = points.unsqueeze(1)
                 S_exp = starts_v.unsqueeze(0)
                 del starts_v
                 D_exp = D_v.unsqueeze(0)
                 del D_v
                 L_exp = L_v.unsqueeze(0)
                 del L_v

                 v_point_start = P_exp - S_exp
                 proj = (v_point_start * D_exp).sum(dim=2)
                 del v_point_start
                 t = torch.clamp(proj, min=torch.tensor(0.0, device=device), max=L_exp)
                 del proj

                 closest_on_axis = S_exp + D_exp * t.unsqueeze(-1)
                 del S_exp
                 del D_exp
                 dist_axis_sq = ((P_exp - closest_on_axis)**2).sum(dim=2)
                 del closest_on_axis
                 frac = t / torch.clamp(L_exp, min=eps)
                 del t
                 del L_exp
                 r0_exp = r0_v.unsqueeze(0)
                 del r0_v
                 r1_exp = r1_v.unsqueeze(0)
                 del r1_v
                 interp_radius = r0_exp * (1.0 - frac) + r1_exp * frac
                 del r0_exp
                 del r1_exp
                 del frac
                 dist_sq_capsule = dist_axis_sq - interp_radius**2
                 del dist_sq_capsule
                 dist_axis = torch.sqrt(torch.clamp(dist_axis_sq, min=0.0))
                 del dist_axis_sq
                 signed_dist_capsule = dist_axis - interp_radius
                 del dist_axis
                 del interp_radius
                 min_dist_capsule, _ = signed_dist_capsule.min(dim=1)
                 del signed_dist_capsule

                 min_dist = torch.minimum(min_dist, min_dist_capsule)
                 del min_dist_capsule
                 # P_exp is still needed for sphere calculation

            del starts, ends, r0, r1, vec, L_sq
            del valid_edge

        if centers.shape[0] > 0:
            # Reuse P_exp if it exists from capsule calculation
            if 'P_exp' not in locals():
                P_exp = points.unsqueeze(1)
            C_exp = centers.unsqueeze(0)
            del centers
            R_exp = radii.unsqueeze(0)
            del radii
            dist_to_centers_sq = ((P_exp - C_exp)**2).sum(dim=2)
            del C_exp
            dist_to_centers = torch.sqrt(torch.clamp(dist_to_centers_sq, min=0.0))
            del dist_to_centers_sq

            signed_dist_sphere = dist_to_centers - R_exp
            del R_exp
            del dist_to_centers
            min_dist_sphere, _ = signed_dist_sphere.min(dim=1)
            del signed_dist_sphere
            min_dist = torch.minimum(min_dist, min_dist_sphere)
            del min_dist_sphere
            del P_exp

        del points
        dist_clamped = torch.clamp(min_dist, max=max_dist)
        return dist_clamped.reshape(*size)