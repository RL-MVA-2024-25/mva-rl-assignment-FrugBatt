import torch
import torch.nn as nn


class DQNModel(nn.Module) :

    def __init__(self):
        """
        DQN module for the HIV patient environment.
        Oberservation space: 6 variables
        Action space: 4 actions (do nothing, provide drug 1, provide drug 2, provide both drugs)
        """
        super(DQNModel, self).__init__()
        self.v = nn.Sequential(
            nn.Linear(6, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        self.a = nn.Sequential(
            nn.Linear(6, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4)
        )

    def forward(self, x):
        v = self.v(x)
        a = self.a(x)
        return v + a - a.mean()
