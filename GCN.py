import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=16)
lrs = [0.001, 0.005, 0.01, 0.05, 0.1]
parser.add_argument('--lr', type=list, default=[0.001, 0.005, 0.01, 0.05, 0.1])
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--wandb', action='store_true', help='Track expeiment')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'GCN-{args.dataset}', lr=args.lr, epochs=args.epochs,
           hidden_channels=args.hidden_channels, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')

from GCN_custom import MyOwnDataset
# dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
dataset = MyOwnDataset(root='.', transform=T.NormalizeFeatures())
data = dataset[0]

for lr in lrs:
    class GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels, cached=True, normalize=False)
            self.conv2 = GCNConv(hidden_channels, out_channels, cached=True, normalize=False)

        def forward(self, x, edge_index, edge_weight=None):
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv1(x, edge_index, edge_weight).relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return x

    model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes)
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=lr)  # Only perform weight-decay on first convolution.


    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr) # Tensor: [2708, 7]
        loss = F.cross_entropy(out[data.train_mask, :], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return float(loss)


    @torch.no_grad()
    def test():
        model.eval()
        pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)  # [2708]
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        return accs


    best_val_acc = final_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
