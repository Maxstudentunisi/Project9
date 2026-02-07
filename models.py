import torch.nn as nn
import torch.nn.functional as F
import torch


class CNNLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 5, padding=2)
        self.c2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.f1 = nn.Linear(64 * 7 * 7, 128)
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.c1(x)))
        x = self.pool(F.relu(self.c2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.f1(x))
        return self.f2(x)


class CNNDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.c4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.f1 = nn.Linear(64 * 7 * 7, 128)
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.pool(x)
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.f1(x))
        return self.f2(x)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(28 * 28, 256)
        self.f2 = nn.Linear(256, 256)
        self.f3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return self.f3(x)


def get_model(name: str):
    name = name.lower()
    if name == "cnn_light":
        return CNNLight()
    if name == "cnn_deep":
        return CNNDeep()
    if name == "mlp":
        return MLP()
    raise ValueError("Modello non valido: usa cnn_light / cnn_deep / mlp")
