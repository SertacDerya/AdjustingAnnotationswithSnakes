import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import numpy as np
from .augmentations import crop
from .utils import load_graph_txt, to_torch

class DRIVEDataset(Dataset):
    
    def __init__(self, train=True, useTrainData=False, cropSize=(96,96), th=15, noise=False, enhancement_factor=10.0):
        
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
            "train": ["/content/drive/MyDrive/windows2/drive/training/mask/{}_training_mask.npy".format(i) for i in range(21,36)],
            "val":  ["/content/drive/MyDrive/windows2/drive/training/mask/{}_training_mask.npy".format(i) for i in range(36,41)]
        }
        
        self.images = image_path["train"] if train else image_path["val"]
        self.labels = label_path["train"] if train else label_path["val"]
        self.masks = masks_path["train"] if train else masks_path["val"]
        self.graphs = graph_path["train"] if train else graph_path["val"]
        
        self.train = train
        self.cropSize = cropSize
        self.th = th
        self.enhancement_factor = enhancement_factor
        self.useTrainData = useTrainData
        
    def __getitem__(self, index):

        image = np.load(self.images[index]).astype(np.float32)
        image = image / 255.0
        label = np.load(self.labels[index]).astype(np.float32)
        mask = np.load(self.masks[index]).astype(np.float32)
        graph = load_graph_txt(self.graphs[index])

        # do the masking for the label here. might create confusion to output 0 values also for the masked spaces
        #binary_mask = (mask == 0)
        #label = label * binary_mask

        for n in graph.nodes:
            graph.nodes[n]["pos"] = graph.nodes[n]["pos"][-1::-1]
            
        slices = None
        
        if self.train:
            image, label, mask, slices = crop([image, label,mask], self.cropSize)
        
        negative_mask = (label == 0)
        label[negative_mask] = self.enhancement_factor    
        label[label>self.th] = self.th
        
        if self.train:
            return torch.tensor(image), torch.tensor(label), torch.tensor(mask), graph, slices
        if self.useTrainData:
            return torch.tensor(image), torch.tensor(label), torch.tensor(mask), graph
        
        return torch.tensor(image), torch.tensor(label), torch.tensor(mask)

    def __len__(self):
        return len(self.images)

def collate_fn(data):
    transposed_data = list(zip(*data))
    images = torch.stack(transposed_data[0], 0).permute(0, 3, 1, 2)
    labels = torch.stack(transposed_data[1], 0).unsqueeze(1)
    masks = torch.stack(transposed_data[2], 0).unsqueeze(1)

    graphs = None
    slices = None
    if len(transposed_data) > 3:
        graphs = transposed_data[3]
    if len(transposed_data) > 4:
        slices = transposed_data[4]

    if graphs is not None:
        if slices is not None:
            return images, labels, masks, graphs, slices
        return images, labels, masks, graphs
    else:
        return images, labels, masks
    
    
