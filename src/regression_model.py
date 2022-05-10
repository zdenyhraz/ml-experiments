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

  def train(self, train_dataset, test_dataset, num_epochs=50, batch_size=16, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    epoch_losses_train = []
    epoch_losses_test = []
    plt.figure()

    for epoch in range(num_epochs):
      epoch_loss_train = 0.0
      for batch_idx_train, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        pred = self.forward(inputs)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        epoch_loss_train += loss.item()
      epoch_loss_train = epoch_loss_train/(batch_idx_train+1)
      epoch_losses_train.append(epoch_loss_train)

      with torch.no_grad():
        epoch_loss_test = 0.0
        for batch_idx_test, (inputs, targets) in enumerate(test_loader):
          optimizer.zero_grad()
          pred = self.forward(inputs)
          loss = criterion(pred, targets)
          epoch_loss_test += loss.item()
        epoch_loss_test = epoch_loss_test/(batch_idx_test+1)
        epoch_losses_test.append(epoch_loss_test)

      #print(f"epoch {epoch} | loss_train: {epoch_loss_train:.3f} | loss_test: {epoch_loss_test:.3f}")

      plt.clf()
      plt.plot(epoch_losses_train, label="train loss")
      plt.plot(epoch_losses_test, label="test loss")
      plt.xlabel("epoch")
      plt.ylabel("loss")
      plt.legend()
      plt.tight_layout()
      plt.pause(1e-9)

  def eval(self, func, train_dataset, x_min=0, x_max=1, n=301):
    with torch.no_grad():
      x = torch.linspace(x_min, x_max, n).reshape(n, -1)
      target = func(x)
      output = self.forward(x)

      train_inputs = []
      train_targets = []
      for batch_inputs, batch_targets in DataLoader(dataset=train_dataset, batch_size=1, shuffle=True):
        for single_input in batch_inputs:
          train_inputs.append(single_input.item())
        for single_target in batch_targets:
          train_targets.append(single_target.item())

      plt.figure()
      plt.plot(x, target, label="target function")
      plt.plot(x, output, label="regression model")
      plt.scatter(train_inputs, train_targets, label="train data", marker="x", c="m", linewidth=2, zorder=2)
      plt.xlabel("input")
      plt.ylabel("output")
      plt.legend()
      plt.tight_layout()
      plt.show()
