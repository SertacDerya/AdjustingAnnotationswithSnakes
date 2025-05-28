import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import numpy as np
from .augmentations import crop
from .utils import load_graph_txt, to_torch
from .Losses.cropGraph import cropGraph_dontCutEdges
from PIL import Image
import os

class DRIVEDataset(Dataset):
    
    def __init__(self, train=True, cropSize=(512, 512), th=15, fit_train=False):
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
        self.fit_train = fit_train
        
    def __getitem__(self, index):
        image = np.load(self.images[index])
        label = np.load(self.labels[index])
        mask = np.load(self.masks[index])
        graph = load_graph_txt(self.graphs[index])
        
        for n in graph.nodes:
            graph.nodes[n]["pos"] = graph.nodes[n]["pos"][-1::-1]
            
        slices = None
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        mask = mask.astype(np.float32)
        
        if self.train and not self.fit_train:
            image, label, mask, slices = crop([image, label, mask], self.cropSize)
            cropped_graph_view = cropGraph_dontCutEdges(graph, slices)
            cropped_graph = cropped_graph_view.copy()

            offset = np.array([s.start for s in slices])
            for n in cropped_graph.nodes:
                if 'pos' in cropped_graph.nodes[n]:
                    cropped_graph.nodes[n]['pos'] = cropped_graph.nodes[n]['pos'] - offset
            
            graph = cropped_graph
            
        label[label>self.th] = self.th
        
        if self.train and not self.fit_train:
            return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), graph, slices
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

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

    if graphs is not None and slices is not None:
        return images, labels, masks, graphs, slices
    else:
        return images, labels, masks