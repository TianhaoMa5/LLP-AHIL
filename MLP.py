import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        return x.div(norm)

class FiveLayerMLP(nn.Module):
    def __init__(self, n_classes, hidden_dim=64, low_dim=32, proj=True):
        super(FiveLayerMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj = proj
        if proj:
            self.l2norm = Normalize(2)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
            self.fc2 = nn.Linear(hidden_dim, low_dim)
            self.fc_last = nn.Linear(low_dim, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        out = x

        if self.proj:
            x = self.fc1(x)
            x = self.relu_mlp(x)
            x = self.fc2(x)
            x = self.l2norm(x)
            out = self.fc_last(x)
        return out

