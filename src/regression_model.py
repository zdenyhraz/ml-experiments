import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class RegressionModel(nn.Module):
  def __init__(self, input_size=1, output_size=1):
    super(RegressionModel, self).__init__()
    self.fc1 = nn.Linear(input_size, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 32)
    self.fc4 = nn.Linear(32, output_size)

  def forward(self, x):
    x = func.relu(self.fc1(x))
    x = func.relu(self.fc2(x))
    x = func.relu(self.fc3(x))
    x = self.fc4(x)  # regression - no relu at the end
    return x

  def train(self, train_dataset):
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 100
    criterion = nn.MSELoss()
    optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
      for batch_idx, (inputs, targets) in enumerate(train_loader):
        pred = self.forward(inputs)
        loss = criterion(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      print(f"epoch {epoch} | loss: {loss.item()}")

  def eval(self, func):
    with torch.no_grad():
      n = 301
      x = torch.linspace(0, 1, n).reshape(n, -1)
      target = func(x)
      output = self.forward(x)

      plt.plot(x, target, label="Target function")
      plt.plot(x, output, label="regression model")
      plt.xlabel("input")
      plt.ylabel("output")
      plt.legend()
      plt.show()
