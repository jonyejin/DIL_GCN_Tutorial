import torch
from pandas import DataFrame
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils import to_undirected
import pandas as pd
import numpy as np
import os.path as osp

# TODO: Fill in this class
class MyOwnDatasetToDo(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['adjacency.tsv', 'cora.content.csv', 'names_labels.tsv']

    @property
    def raw_dir(self) -> str:
        return 'Data Files'

    @property
    def processed_file_names(self):
        return ['data.pt']

    @staticmethod
    def load_edge_index_from_tsv(path='Data Files/adjacency.tsv', is_undirected=True):
        """
        :param path: tsv 파일이 있는 위치
        :param is_undirected: undirected 라면 edge_index를 to_undirected함수를 통해서 바꿔줘야 한다.
        :return: edge_index를 리턴해준다.(tensor shape: [2, edge수]
        """


    def process(self):
        """
        Data 객체를 만들어 준다.
        :return: None
        """



m = MyOwnDatasetToDo('.')
