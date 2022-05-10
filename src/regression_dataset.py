import torch
from torch.utils.data import Dataset
import numpy as np


class RegressionDataset(Dataset):

  def __init__(self, func, n, x_min=0, x_max=1):
    print(f"Initializing RegressionDataset (n={n})")
    self.inputs = []
    self.targets = []
    for idx in range(n):
      x = x_min+(x_max-x_min)*torch.rand(1)
      y = func(x)
      self.inputs.append(x)
      self.targets.append(y)

    print(
        f"Regression dataset created ({len(self.inputs)} {self.inputs[0].shape} inputs, {len(self.targets)} {self.targets[0].shape} targets)")

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx):
    return (self.inputs[idx], self.targets[idx])
