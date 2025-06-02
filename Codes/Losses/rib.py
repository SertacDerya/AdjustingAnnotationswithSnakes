from .snake import Snake
import torch
import torch.nn.functional as F
import math
from .cropGraph import cropGraph_dontCutEdges
from .renderDistanceMap import getCropCoords


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
        self.w = torch.ones(self.s.shape[0]) * 8.0
        #self.w = torch.randint(low=2, high=10, size=(self.s.shape[0],), dtype=torch.float32)

    def cuda(self):
        super().cuda()
        # move the widths to gpu
        self.w = self.w.cuda()

    def set_w(self, widths):
        self.w = widths

    def get_w(self):
        return self.w

    def _compute_normals(self, pos_tensor): # Renamed pos to pos_tensor to avoid conflict
        N, d = pos_tensor.shape
        eps_val = 1e-8 
        
        tangents = torch.zeros_like(pos_tensor)

        if N == 0: # Handle empty snake
             if d == 2: return (torch.zeros_like(pos_tensor),)
             else: return (torch.zeros_like(pos_tensor), torch.zeros_like(pos_tensor), torch.zeros_like(pos_tensor))

        for k_idx in range(N):
            node_name = self.i2n[k_idx]
            current_node_pos = pos_tensor[k_idx]
            
            neighbor_node_names = list(self.h.neighbors(node_name)) # Use graph neighbors
            
            t_k = torch.zeros(d, device=pos_tensor.device, dtype=pos_tensor.dtype)

            if len(neighbor_node_names) == 1:
                neigh_idx = self.n2i[neighbor_node_names[0]]
                neigh_pos = pos_tensor[neigh_idx]
                t_k = neigh_pos - current_node_pos
            elif len(neighbor_node_names) >= 2:
                neighbor_indices = [self.n2i[name] for name in neighbor_node_names]
                
                best_na_pos, best_nb_pos = None, None
                min_dot_product = float('inf')

                if len(neighbor_indices) == 2:
                    best_na_pos = pos_tensor[neighbor_indices[0]]
                    best_nb_pos = pos_tensor[neighbor_indices[1]]
                else: # Branch point: select two "most collinear" neighbors
                    for i in range(len(neighbor_indices)):
                        for j in range(i + 1, len(neighbor_indices)):
                            ni_pos = pos_tensor[neighbor_indices[i]]
                            nj_pos = pos_tensor[neighbor_indices[j]]
                            
                            vec_p_ni = ni_pos - current_node_pos
                            vec_p_nj = nj_pos - current_node_pos
                            
                            norm_p_ni = vec_p_ni.norm()
                            norm_p_nj = vec_p_nj.norm()

                            if norm_p_ni > eps_val and norm_p_nj > eps_val:
                                dot_prod = torch.dot(vec_p_ni / norm_p_ni, vec_p_nj / norm_p_nj)
                                if dot_prod < min_dot_product:
                                    min_dot_product = dot_prod
                                    best_na_pos = ni_pos
                                    best_nb_pos = nj_pos
                
                if best_na_pos is not None and best_nb_pos is not None:
                    t_k = best_nb_pos - best_na_pos # Tangent along line connecting these two "opposite" neighbors
            
            tangents[k_idx] = t_k

        tangent_lengths = tangents.norm(dim=1, keepdim=True)
        tangents = tangents / (tangent_lengths + eps_val) # Normalize all tangents

        if d == 2:
            normals_calc = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=1)
            return (normals_calc,) # Tuple with one element: normals tensor
        elif d == 3:
            a = torch.zeros_like(tangents)
            a[:, 0] = 1.0 
            parallel_mask = (tangents * a).abs().sum(dim=1) > 0.99 
            a[parallel_mask, 0] = 0.0
            a[parallel_mask, 1] = 1.0

            n1 = torch.cross(tangents, a, dim=1)
            n1 = n1 / (n1.norm(dim=1, keepdim=True) + eps_val)
            n2 = torch.cross(tangents, n1, dim=1)
            n2 = n2 / (n2.norm(dim=1, keepdim=True) + eps_val) # Re-normalize for stability
            return (n1, n2, tangents) # Tuple: n1, n2, tangents
        else:
            raise ValueError(f"Unsupported dimension for normals: {d}")
        
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
        points = torch.stack([m.flatten() for m in mesh], dim=1)
        num_points = points.shape[0]
        min_dist = torch.full((num_points,), float('inf'), device=device, dtype=centers.dtype)

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

            vec = ends - starts
            L_sq = (vec**2).sum(dim=1)
            valid_edge = L_sq > eps**2
            if torch.any(valid_edge):
                 starts_v, ends_v = starts[valid_edge], ends[valid_edge]
                 r0_v, r1_v = r0[valid_edge], r1[valid_edge]
                 vec_v = vec[valid_edge]
                 L_sq_v = L_sq[valid_edge]
                 L_v = torch.sqrt(L_sq_v + eps)
                 D_v = vec_v / (L_v.unsqueeze(1) + eps)

                 P_exp = points.unsqueeze(1)
                 S_exp = starts_v.unsqueeze(0)
                 D_exp = D_v.unsqueeze(0)
                 L_exp = L_v.unsqueeze(0)

                 v_point_start = P_exp - S_exp
                 proj = (v_point_start * D_exp).sum(dim=2)
                 t = torch.clamp(proj, min=torch.tensor(0.0, device=device), max=L_exp)

                 closest_on_axis = S_exp + D_exp * t.unsqueeze(-1)
                 dist_axis_sq = ((P_exp - closest_on_axis)**2).sum(dim=2)
                 frac = t / torch.clamp(L_exp, min=eps)
                 r0_exp = r0_v.unsqueeze(0)
                 r1_exp = r1_v.unsqueeze(0)
                 interp_radius = r0_exp * (1.0 - frac) + r1_exp * frac
                 dist_axis = torch.sqrt(torch.clamp(dist_axis_sq, min=eps))
                 signed_dist_capsule = dist_axis - interp_radius
                 min_dist_capsule, _ = signed_dist_capsule.min(dim=1)

                 min_dist = torch.minimum(min_dist, min_dist_capsule)
                 # P_exp is still needed for sphere calculation

        if centers.shape[0] > 0:
            # Reuse P_exp if it exists from capsule calculation
            if 'P_exp' not in locals():
                P_exp = points.unsqueeze(1)
            C_exp = centers.unsqueeze(0)
            R_exp = radii.unsqueeze(0)
            dist_to_centers_sq = ((P_exp - C_exp)**2).sum(dim=2)
            dist_to_centers = torch.sqrt(torch.clamp(dist_to_centers_sq, min=eps))

            signed_dist_sphere = dist_to_centers - R_exp
            min_dist_sphere, _ = signed_dist_sphere.min(dim=1)
            min_dist = torch.minimum(min_dist, min_dist_sphere)

        dist_clamped = torch.clamp(min_dist, max=max_dist)
        return dist_clamped.reshape(*size)
    
    def render_distance_map_with_widths_cropped(self, size, cropsz, max_dist=16.0, max_edge_len=None):
        """
        Unified 2D/3D signed distance map for the ribbon snake using graph structure,
        rendered in crops to manage memory.

        Args:
            size (tuple): (W, H) for 2D or (X, Y, Z) for 3D grid dimensions of the full map.
            cropsz (tuple): (cW, cH) or (cX, cY, cZ) for crop dimensions.
            max_dist (float): Maximum distance value to clamp to.
            max_edge_len (float, optional): Maximum expected edge length in the graph.
                                            Used to refine margin calculation for graph cropping.

        Returns:
            torch.Tensor: Signed distance map of shape `size`. Negative inside,
                          zero on surface, positive outside up to max_dist.
        """
        device = self.s.device
        full_centers = self.s
        full_radii = (self.w.flatten() / 2.0) # widths divided by 2
        eps = 1e-8

        if full_centers.numel() == 0 or full_radii.numel() == 0 or not self.h or len(self.h.nodes) == 0:
            print("Warning: Rendering distance map for empty snake.")
            return torch.full(size, max_dist, device=device, dtype=full_centers.dtype)

        if full_centers.shape[0] != full_radii.shape[0]:
             raise ValueError(f"Mismatch between center points ({full_centers.shape[0]}) and radii ({full_radii.shape[0]})")

        output_dist_map = torch.full(size, max_dist, device=device, dtype=full_centers.dtype)
        max_r_snake = full_radii.max().item() if full_radii.numel() > 0 else 0.0
        
        margin_for_graph = max_dist + max_r_snake
        if max_edge_len is not None and max_edge_len > 0:
            margin_for_graph = (margin_for_graph + 0.5 * max_edge_len) / (2.0**0.5)
        
        margin_for_graph = max(margin_for_graph, 0)

        graph_crop_slices_list = getCropCoords(size, cropsz, margin_for_graph)
        distmap_crop_slices_list = getCropCoords(size, cropsz, 0)

        for g_slices, d_slices in zip(graph_crop_slices_list, distmap_crop_slices_list):
            current_crop_size_tuple = tuple(s.stop - s.start for s in d_slices)
            if any(cs <= 0 for cs in current_crop_size_tuple):
                continue
            cropped_graph = cropGraph_dontCutEdges(self.h, g_slices)


            if not cropped_graph or not cropped_graph.nodes:
                continue

            crop_axes = [torch.arange(s.start, s.stop, device=device, dtype=torch.float32) for s in d_slices]
            if any(len(ax) == 0 for ax in crop_axes):
                continue
            
            mesh = torch.meshgrid(*crop_axes, indexing='ij')
            points_crop = torch.stack([m.flatten() for m in mesh], dim=1)
            num_points_crop = points_crop.shape[0]

            if num_points_crop == 0:
                continue

            min_dist_for_crop = torch.full((num_points_crop,), max_dist, device=device, dtype=full_centers.dtype)
            
            P_exp_crop = points_crop.unsqueeze(1)

            if len(cropped_graph.edges) > 0:
                edge_indices_list_c = []
                try:
                    for u_name, v_name in cropped_graph.edges:
                        if u_name in self.n2i and v_name in self.n2i:
                            edge_indices_list_c.append((self.n2i[u_name], self.n2i[v_name]))
                        else:
                            print(f"Warning: Node {u_name} or {v_name} from cropped_graph edge not in self.n2i. Skipping edge.")
                except KeyError as e:
                    raise RuntimeError(f"Node ID {e} from graph edges not found in n2i mapping.")

                if edge_indices_list_c:
                    edge_indices_c = torch.tensor(edge_indices_list_c, device=device, dtype=torch.long)
                    
                    starts_c = full_centers[edge_indices_c[:, 0]]
                    ends_c   = full_centers[edge_indices_c[:, 1]]  
                    r0_c     = full_radii[edge_indices_c[:, 0]]  
                    r1_c     = full_radii[edge_indices_c[:, 1]]  

                    vec_c = ends_c - starts_c
                    L_sq_c = (vec_c**2).sum(dim=1)
                    valid_edge_mask_c = L_sq_c > eps**2

                    if torch.any(valid_edge_mask_c):
                        starts_v = starts_c[valid_edge_mask_c]
                        r0_v = r0_c[valid_edge_mask_c]
                        r1_v = r1_c[valid_edge_mask_c]
                        vec_v = vec_c[valid_edge_mask_c]
                        
                        L_v = torch.sqrt((vec_v**2).sum(dim=1) + eps)
                        D_v = vec_v / (L_v.unsqueeze(1) + eps)     
                    
                        S_exp = starts_v.unsqueeze(0)
                        D_exp = D_v.unsqueeze(0)     
                        L_exp = L_v.unsqueeze(0)     
                    
                        v_point_start = P_exp_crop - S_exp                  
                        proj = (v_point_start * D_exp).sum(dim=2)           
                        t = torch.clamp(proj, min=torch.tensor(eps, device=device), max=L_exp)           

                        closest_on_axis = S_exp + D_exp * t.unsqueeze(-1)   
                        dist_axis_sq = ((P_exp_crop - closest_on_axis)**2).sum(dim=2)
                        dist_axis = torch.sqrt(torch.clamp(dist_axis_sq, min=torch.tensor(eps,device=device)))   
                        
                        frac = t / torch.clamp(L_exp, min=torch.tensor(eps,device=device))              
                        r0_exp = r0_v.unsqueeze(0)                          
                        r1_exp = r1_v.unsqueeze(0)                          
                        interp_radius = r0_exp * (1.0 - frac) + r1_exp * frac
                        
                        signed_dist_capsule = dist_axis - interp_radius     
                        
                        if signed_dist_capsule.numel() > 0:
                            min_dist_capsule, _ = signed_dist_capsule.min(dim=1)
                            min_dist_for_crop = torch.minimum(min_dist_for_crop, min_dist_capsule)

            node_indices_list_c = []
            try:
                for node_name in cropped_graph.nodes:
                    if node_name in self.n2i:
                        node_indices_list_c.append(self.n2i[node_name])
                    else:
                        print(f"Warning: Node {node_name} from cropped_graph.nodes not in self.n2i. Skipping for sphere.")
            except KeyError as e:
                 raise RuntimeError(f"Node ID {e} from graph nodes not found in n2i mapping.")

            if node_indices_list_c:
                node_indices_c = torch.tensor(node_indices_list_c, device=device, dtype=torch.long)
            
                current_centers_c = full_centers[node_indices_c]
                current_radii_c = full_radii[node_indices_c]    

                if current_centers_c.numel() > 0:
                    C_exp = current_centers_c.unsqueeze(0)
                    R_exp = current_radii_c.unsqueeze(0)  
                
                    dist_to_centers_sq = ((P_exp_crop - C_exp)**2).sum(dim=2)
                    dist_to_centers = torch.sqrt(torch.clamp(dist_to_centers_sq, min=torch.tensor(eps,device=device)))

                    signed_dist_sphere = dist_to_centers - R_exp         
                    
                    if signed_dist_sphere.numel() > 0:
                        min_dist_sphere, _ = signed_dist_sphere.min(dim=1)   
                        min_dist_for_crop = torch.minimum(min_dist_for_crop, min_dist_sphere)
        
            dist_clamped_crop = torch.clamp(min_dist_for_crop, max=max_dist)
            output_dist_map[d_slices] = dist_clamped_crop.reshape(*current_crop_size_tuple)

        return output_dist_map