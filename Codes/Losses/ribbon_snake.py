from .snake import Snake
import torch
from torch import nn
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as dist
from . import gradImSnake
from .renderDistanceMap import getCropCoords
from .renderLineGraph import drawLine


class Ribbon_Snake(Snake):
# represents a class that is similar to snake but also contains width information
# the width information can be represented by sampling along the nodes of the snake and
# attaching width information to the samples
# each width will try to minimize a energy
# the energy they will try to minimize will try to fit the width to the edge of the distance map
# will try different values to fit the
#### Assumptions
# the shape of the 3d structure is tought of as a cylinder that is a perfect circle
# this can be changed to aa elypse like structure too, but not implemented yet 
    def __init__(self,graph,crop,stepsz,alpha,beta,ndims, num_samples=10):
    # added a way to select how many samples to take between each control point
    # for each sample we keep its width info
    # at the start all of the sample are set to width 1
        super(Ribbon_Snake,self).__init__(graph,crop,stepsz,alpha,beta,ndims)
        self.num_samples = num_samples
        self.samples = self.create_samples()
        self.widths = np.ones((len(self.samples),))
        self.iscuda = False

    def cuda(self):
        super(Ribbon_Snake, self).cuda()
        self.iscuda = True

    def create_samples(self):
    # we look for the each edge of the graph
    # then we sample the points between in equal intervals
    # return their possitions
        samples = []
        for u, v in self.h.edges:
            pos_u = self.h.nodes[u]["pos"]
            pos_v = self.h.nodes[v]["pos"]
            for t in np.linspace(0, 1, self.num_samples, endpoint=False):
                sample_pos = (1 - t) * pos_u + t * pos_v
                samples.append(sample_pos)
        return np.array(samples)
    
    def set_widths(self, widths):
    # may need more complicated way to set the widths
        self.widths = widths

    def get_samples_with_widths(self):
    # returns the samples along with their widths
        return list(zip(self.samples, self.widths))
    
    def render_width_map(self, size, cropsz, dmax, maxedgelen):
    # this tries to create a map that represents the ribbon snake
    # the parts in the map belonging to the snake will be 1 and others 0
    # centerline will be determines by the node locations
    # and the width will be determine by the sample's width info
    # we will try to create a cylinder from the cetnerline and sample width information
    # inside the cylinder will be set to 1s
    ## this will be used in the loss function to measure models performance in
    # predicting the overall width and centerline info
        width_map = torch.zeros(size, dtype=torch.float32, device='cuda' if self.iscuda else 'cpu')
    
        samples_with_widths = self.get_samples_with_widths()
        for idx, (sample, width) in enumerate(samples_with_widths):
            sample_idx = tuple(map(int, np.round(sample)))
            radius = int(np.ceil(width / 2))

            # Draw the circle for the current sample
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if i**2 + j**2 <= radius**2:
                        x, y = sample_idx[0] + i, sample_idx[1] + j
                        if 0 <= x < size[0] and 0 <= y < size[1]:
                            width_map[x, y] = 1.0

            # Connect the samples with a line and draw the cylinder
            if idx < len(samples_with_widths) - 1:
                next_sample = samples_with_widths[idx + 1][0]
                next_sample_idx = tuple(map(int, np.round(next_sample)))
                line_coords = drawLine(np.zeros(size), np.array(sample_idx), np.array(next_sample_idx))
                for lx, ly in line_coords:
                    if 0 <= lx < size[0] and 0 <= ly < size[1]:
                        width_map[lx, ly] = 1.0

                min_x = min(sample_idx[0] - radius, next_sample_idx[0] - radius)
                max_x = max(sample_idx[0] + radius, next_sample_idx[0] + radius)
                min_y = min(sample_idx[1] - radius, next_sample_idx[1] - radius)
                max_y = max(sample_idx[1] + radius, next_sample_idx[1] + radius)

                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        if 0 <= x < size[0] and 0 <= y < size[1]:
                            dist_curr = (x - sample_idx[0])**2 + (y - sample_idx[1])**2
                            dist_next = (x - next_sample_idx[0])**2 + (y - next_sample_idx[1])**2

                            on_line = False
                            line_coords_fill = drawLine(np.zeros(size), np.array(sample_idx), np.array(next_sample_idx))
                            for lxf, lyf in line_coords_fill:
                                if lxf == x and lyf == y:
                                    on_line = True
                                    break

                            if dist_curr <= radius**2 or dist_next <= radius**2 or on_line:
                                width_map[x, y] = 1.0

        return width_map
    

class RibonFastLoss(nn.Module):
    # the main idea for this loss will be to put the width in an optimal place
    # according to the probability map given by the model
    # we will treat the probability with the highest point as the center like normally
    # but we will also treat points above a certain threshold as points inside the
    # linear structure
    # then we will try to set the widths for the samples to align with the probability map's results
    # but we will also try to not make the width too big
    # need to decide how to optimize this
    def __init__(self, stepsz,alpha,beta,fltrstdev,ndims,nsteps,
                       cropsz,dmax,maxedgelen,extgradfac):
        super(RibonFastLoss,self).__init__()
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

        self.fltr = gradImSnake.makeGaussEdgeFltr(self.fltrstdev,self.ndims)
        self.fltrt = torch.from_numpy(self.fltr).type(torch.float32)

        self.iscuda = False

    def cuda(self):
        super(RibonFastLoss,self).cuda()
        self.fltrt = self.fltrt.cuda()
        self.iscuda = True
        return self

    def forward(self,pred_dmap,lbl_graphs,crops=None):
    # how to get the probability map?
    # pred_dmap is the predicted distance map from the UNet
    # lbl_graphs contains graphs each represent a label as a snake (not exactly a snake but a graph which represents a snake) / not snake class
    # crops is a list of slices, each represents the crop area of the corresponding snake

        pred_ = pred_dmap
        gimg = gradImSnake.cmptGradIm(pred_,self.fltrt)
        gimg *= self.extgradfac
        snake_dmap = []
        ribbon_probmap = []

        for i,lg in enumerate(zip(lbl_graphs,gimg)):
            # i is index num
            # lg is a tuple of a graph and a gradient image
            l = lg[0] # graph
            g = lg[1] # gradient image

            if crops:
                crop = crops[i]
            else:
                crop=[slice(0,s) for s in g.shape[1:]]
            s = gradImSnake.GradImSnake(l,crop,self.stepsz,self.alpha,
                                      self.beta,self.ndims,g)
            if self.iscuda: s.cuda()

            s.optim(self.nsteps)

            # getting a probability map in some way
            # if pred_map is the distance we can use a sigmoid of sosme sorts to get the probability map
            # but smaller values in distance map needs to have higher probability
            prob_map = torch.sigmoid(-pred_dmap[i])
            samples_with_widths = s.get_samples_with_widths()

            # Calculate widths for each sample
            new_widths = []
            threshold = 0.5  # Define a probability threshold
            for sample, _ in samples_with_widths:
                sample_idx = tuple(map(int, np.round(sample)))
                radius = 0
                found = False
                for r in range(1, g.shape[1]):
                    for angle in range(0, 360, 10):
                        # check for the furthest point that is still above the prob thres
                        x = int(sample_idx[0] + r * np.cos(np.radians(angle)))
                        y = int(sample_idx[1] + r * np.sin(np.radians(angle)))
                        if 0 <= x < g.shape[1] and 0 <= y < g.shape[2]:
                            if prob_map[x, y].item() < threshold:
                                radius = r - 1
                                found = True
                                break
                        else:
                            found = True
                            continue
                    if found:
                        break
                new_widths.append(radius)

            s.set_widths(np.array(new_widths))

            pmap = s.render_width_map(g.shape[1:],self.cropsz,self.dmax,
                                     self.maxedgelen)
            
            ribbon_probmap.append(pmap)

            dmap = s.renderDistanceMap(g.shape[1:],self.cropsz,self.dmax,
                                     self.maxedgelen)
            snake_dmap.append(dmap)

        # because of the width information, need an additinal loss to also
        # calculate the loss connected to under or over estimation
        # render another map with width info, from the ribbon snake
        # and also calculate a loss for this
        # this just looks at the distance map
        snake_dm = torch.stack(snake_dmap,0).unsqueeze(1)
        loss = torch.pow(pred_dmap-snake_dm,2).mean()

        ######
        # something like this
        ######
        # width_loss = torch.pow(prob_map-snake_dm,2).mean()
        # loss = loss + width_loss
        # also we can weigh the width loss differently to reduce or increase its effect
        ribbon_pm = torch.stack(ribbon_probmap,0).unsqueeze(1)
        loss += torch.pow(torch.sigmoid(pred_dmap)-ribbon_pm,2).mean()
        ######
                  
        self.snake = s
        self.gimg = gimg
        
        return loss
