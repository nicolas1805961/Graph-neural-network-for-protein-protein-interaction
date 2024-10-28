import torch
import torch.nn.functional as F
from torch_geometric.nn import Sequential, GCNConv, global_mean_pool
from torch import nn

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedder = nn.Embedding(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = Sequential('x, edge_index', [
                       (GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'),
                       nn.ReLU(inplace=True),
                       nn.BatchNorm1d(hidden_dim)])
            self.layers.append(layer)

    def forward(self, x, edge_index, batch):

        x = x.squeeze()

        x = self.embedder(x)

        for layer in self.layers:
            x = layer(x=x, edge_index=edge_index)

        x = global_mean_pool(x, batch)

        return x

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, data):

        emb_1 = self.encoder(x=data.x_1, edge_index=data.edge_index_1, batch=data.x_1_batch)
        emb_2 = self.encoder(x=data.x_2, edge_index=data.edge_index_2, batch=data.x_2_batch)

        cated = torch.cat([emb_1, emb_2], dim=1)

        out = self.mlp(cated)

        return out