import torch
from pandas import DataFrame
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils import to_undirected
import pandas as pd
import numpy as np
import os.path as osp


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['adjacency.tsv', 'cora.content.csv', 'names_labels.tsv']

    @property
    def raw_dir(self) -> str:
        return 'CustomCora'

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
    #     ...

    @staticmethod
    def load_edge_index_from_tsv(path='CustomCora/adjacency.tsv', is_undirected=True):
        """
        @param node_names: name of nodes(Integer)
        @return:[Tensor] (shape: [2, num_edges])
        """

        df = pd.read_csv(path, '\t', header=None, names=["node1", "node2"], usecols=["node1", "node2"])
        edge_index = torch.from_numpy(df.to_numpy())
        edge_index = edge_index.t()
        if is_undirected:
            # generate other direction of edge index
            edge_index = to_undirected(edge_index)
        return edge_index

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv('CustomCora/labels.tsv', header=None)
        _y = torch.from_numpy(df.to_numpy().astype('float32')).squeeze().type(torch.LongTensor)
        feature_vectors = pd.read_csv('CustomCora/cora.content.csv', '\t', header=None).iloc[:, 1:-1]
        _x = torch.from_numpy(feature_vectors.to_numpy().astype('float32'))

        data = Data(x=_x, edge_index=self.load_edge_index_from_tsv(), y=_y)
        # undirected feature

        # train, val, test
        train_masks, val_masks, test_masks = [], [], []
        # for i in range(10):
        name = f'cora_split_0.6_0.2_{0}.npz'
        splits = np.load(osp.join(self.raw_dir, name))

        data.train_mask = torch.from_numpy(splits['train_mask'])
        data.val_mask = torch.from_numpy(splits['val_mask'])
        data.test_mask = torch.from_numpy(splits['test_mask'])

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


m = MyOwnDataset('.')
print("??/")
