from torch_geometric.data import Data, Dataset
import numpy as np
import torch
import os
import pandas as pd

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_1':
            return self.x_1.size(0)
        if key == 'edge_index_2':
            return self.x_2.size(0)
        return super().__inc__(key, value, *args, **kwargs)
    

class MyOwnDataset(Dataset):
    def __init__(self, root, file_name, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.df = pd.read_csv(os.path.join(root, file_name), header=0, encoding='ISO-8859-1')
        self.root = root
        
    def len(self):
        return len(self.df)  # Length based on raw .npy files

    def get(self, idx):
        row = self.df.iloc[idx]

        one_hot_1 = np.load(os.path.join(self.root, 'npy', 'one_hot', row['protein_xref_1'] + '.npy'))
        one_hot_2 = np.load(os.path.join(self.root, 'npy', 'one_hot', row['protein_xref_2'] + '.npy'))

        adjacency_1 = np.load(os.path.join(self.root, 'npy', 'adjacency', row['protein_xref_1'] + '.npy'))
        adjacency_2 = np.load(os.path.join(self.root, 'npy', 'adjacency', row['protein_xref_2'] + '.npy'))

        one_hot_1 = torch.from_numpy(one_hot_1)
        one_hot_2 = torch.from_numpy(one_hot_2)

        adjacency_1 = torch.from_numpy(adjacency_1)
        adjacency_2 = torch.from_numpy(adjacency_2)

        data = PairData(x_1=one_hot_1, edge_index_1=adjacency_1, x_2=one_hot_2, edge_index_2=adjacency_2)
        
        return data