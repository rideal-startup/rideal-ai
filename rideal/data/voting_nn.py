from typing import List, Tuple

import torch
import syft as sy

import numpy as np
import pandas as pd

hook = sy.TorchHook(torch)

IDX_2_LABEL = ['Still', 'Car', 'Train', 'Bus', 'Walking']
LABEL_2_IDX = {l: i for i, l in enumerate(IDX_2_LABEL)}


def load_data(path: str):
    df = pd.read_csv(path)
    x = df.drop(['target', 'time'], axis='columns').values.astype('float32')
    y = df['target'].values 
    y = torch.LongTensor([LABEL_2_IDX[l] for l in y])

    return torch.from_numpy(x), y


def create_workers(n_workers: int = 3):
    return [sy.VirtualWorker(hook, id=str(i)) 
            for i in range(n_workers)]


def distribute_data(workers: List[sy.VirtualWorker], 
                    x: torch.Tensor, 
                    y: torch.Tensor) -> List[Tuple[sy.PointerTensor, 
                                                   sy.PointerTensor]]:
    
    n_splits = len(workers)
    n_elems = x.size(0)
    idx_splits = (torch.randperm(n_elems).long()
                  .split(n_elems // n_splits, dim=0)[:n_splits])
    
    datasets = []
    for w, s in zip(workers, idx_splits):
        datasets.append((x[s].send(w), y[s].send(w)))

    return datasets

