import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 900),
        )

    def forward(self, x):
        return self.network(x)


class EnsembleMLPModel(nn.Module):
    """Wraps multiple trained MLPModel instances and combines their predictions."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        n = len(members)
        self.register_buffer("weights", torch.ones(n) / n)

    def set_weights(self, weights):
        """Set member weights (must sum to 1)."""
        self.weights = torch.tensor(weights, dtype=torch.float32, device=self.weights.device)

    def forward(self, x):
        outputs = torch.stack([member(x) for member in self.members])
        return (outputs * self.weights.view(-1, 1, 1)).sum(dim=0)
