import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, n_features, n_labels):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(n_features, 32)
        self.dropout = nn.Dropout()
        self.linear2 = nn.Linear(32, n_labels)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x