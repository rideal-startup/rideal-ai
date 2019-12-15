import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):

    def __init__(self, n_features, n_labels):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(n_features, 1024)

        self.linear2 = nn.Linear(1024, 512)

        self.linear3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(.5)

        self.linear4 = nn.Linear(256, 128)
        self.dropout4 = nn.Dropout(.5)

        self.clf = nn.Linear(128, n_labels)

    def forward(self, x):
        x = F.relu(self.linear1(x))

        x = F.relu(self.linear2(x))

        x = F.relu(self.linear3(x))
        x = self.dropout3(x)

        x = F.relu(self.linear4(x))
        x = self.dropout4(x)

        x = self.clf(x)

        return x
