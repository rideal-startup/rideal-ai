import torch
import pandas as pd


class TMDDataset(object):

    def __init__(self, path: str):
        self.df = pd.read_csv(path)
        self.df = self.df.drop('time', axis='columns')
        self.idx_to_label = self.df['target'].unique()
        self.to_idx = {l: i for i, l in enumerate(self.idx_to_label)}
        
    def __getitem__(self, idx):
        instance = self.df.iloc[idx, :]
        x = instance.drop('target')
        y = instance['target']
        y = self.to_idx[y]
        return torch.FloatTensor(x), torch.LongTensor([y])

    def __len__(self):
        return self.df.shape[0]

