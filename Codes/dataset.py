import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import numpy as np
from .augmentations import crop
from .utils import load_graph_txt, to_torch
from PIL import Image
import os

class SynthDataset(Dataset):
    
    def __init__(self, train=True, cropSize=(96,96,96), th=15, noise=False):
        
        image_path = {
            "train": ["/content/drive/MyDrive/synth_data/images/data_{}.npy".format(i) for i in range(25)],
            "val":  ["/content/drive/MyDrive/synth_data/images/data_{}.npy".format(i) for i in range(25,30)]
        }
        label_path = {
            # Using noise_2_dist_labels instead of dist_labels
            "train": ["/content/drive/MyDrive/synth_data/noise_2_dist_labels/data_{}.npy".format(i) for i in range(25)],
            "noise": ["/content/drive/MyDrive/synth_data/noise_2_dist_labels/data_{}.npy".format(i) for i in range(25)],
            "val":  ["/content/drive/MyDrive/synth_data/noise_2_dist_labels/data_{}.npy".format(i) for i in range(25,30)]
        }
        graph_path = {
            "train": ["/content/drive/MyDrive/synth_data/graphs/data_{}.graph".format(i) for i in range(25)],
            "noise": ["/content/drive/MyDrive/synth_data/noise_2_graphs/data_{}.graph".format(i) for i in range(25)],
            "val":  ["/content/drive/MyDrive/synth_data/graphs/data_{}.graph".format(i) for i in range(25,30)]
        }
        
        self.images = image_path["train"] if train else image_path["val"]
        self.labels = label_path["train"] if train and not noise else label_path["noise"] if train and noise else label_path["val"]
        self.graphs = graph_path["train"] if train and not noise else graph_path["noise"] if train and noise else graph_path["val"]
            
        self.train = train
        self.cropSize = cropSize 
        self.th = th
        
    def __getitem__(self, index):

        image = np.load(self.images[index])
        label = np.load(self.labels[index])
        graph = load_graph_txt(self.graphs[index])
        
        for n in graph.nodes:
            graph.nodes[n]["pos"] = graph.nodes[n]["pos"][-1::-1]
            
        slices = None
        
        if self.train:
            image, label, slices = crop([image, label], self.cropSize)
            
        label[label>self.th] = self.th
        
        if self.train:
            return torch.tensor(image), torch.tensor(label), graph, slices
        
        return torch.tensor(image), torch.tensor(label)

    def __len__(self):
        return len(self.images)


class DRIVEDataset(Dataset):
    
    def __init__(self, train=True, cropSize=(512, 512), th=15):
        image_path = {
            "train": ["/content/drive/MyDrive/windows2/drive/training/images/{}_training.npy".format(i) for i in range(21,36)],
            "val":  ["/content/drive/MyDrive/windows2/drive/training/images/{}_training.npy".format(i) for i in range(36,41)]
        }
        label_path = {
            # Using noise_2_dist_labels instead of dist_labels
            "train": ["/content/drive/MyDrive/windows2/drive/training/inverted_labels/{}_manual1.npy".format(i) for i in range(21,36)],
            "val":  ["/content/drive/MyDrive/windows2/drive/training/inverted_labels/{}_manual1.npy".format(i) for i in range(36,41)]
        }
        graph_path = {
            "train": ["/content/drive/MyDrive/windows2/drive/training/graphs/{}_manual1.npy.graph".format(i) for i in range(21,36)],
            "val":  ["/content/drive/MyDrive/windows2/drive/training/graphs/{}_manual1.npy.graph".format(i) for i in range(36,41)]
        }
        # i guess the mask will be applied after unets prediction su that unnecessary info is blocked
        masks_path = {
            "train": ["/content/drive/MyDrive/windows2/drive/training/mask/{}_training_mask.gif".format(i) for i in range(21,36)],
            "val":  ["/content/drive/MyDrive/windows2/drive/training/mask/{}_training_mask.gif".format(i) for i in range(36,41)]
        }
        
        self.images = image_path["train"] if train else image_path["val"]
        self.labels = label_path["train"] if train else label_path["val"]
        self.masks = image_path["train"] if train else image_path["val"]
        self.graphs = graph_path["train"] if train else graph_path["val"]
        
        self.train = train
        self.cropSize = cropSize
        self.th = th
        
    def __getitem__(self, index):
        image = np.load(self.images[index])
        label = np.load(self.labels[index])
        mask = np.load(self.masks[index])
        graph = load_graph_txt(self.graphs[index])
        
        """ for n in graph.nodes:
            graph.nodes[n]["pos"] = graph.nodes[n]["pos"][-1::-1] """
            
        slices = None
        
        if self.train:
            image, label, mask, slices = crop([image, label, mask], self.cropSize)
            
        label[label>self.th] = self.th
        
        if self.train:
            return torch.tensor(image), torch.tensor(label), torch.tensor(mask), graph, slices
        
        return torch.tensor(image), torch.tensor(label), torch.tensor(mask)

    def __len__(self):
        return len(self.images)


def collate_fn(data):
    transposed_data = list(zip(*data))
    images = torch.stack(transposed_data[0], 0)[:,None]
    labels = torch.stack(transposed_data[1], 0)[:,None]
    graphs = transposed_data[2]
    slices = transposed_data[3]
    
    return images, labels, graphs, slices

def drive_collate_fn(data):
    transposed_data = list(zip(*data))
    images = torch.stack(transposed_data[0], 0)[:,None]
    labels = torch.stack(transposed_data[1], 0)[:,None]
    
    if len(transposed_data) > 2:
        graphs = transposed_data[2]
        slices = transposed_data[3]
        return images, labels, graphs, slices
    
    return images, labels