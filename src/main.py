import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import matplotlib.pyplot as plt


class Net(nn.Module):
  def __init__(self, input_size=1, output_size=1):
    super(Net, self).__init__()
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


def TargetFunction(x):
  return torch.exp(-20. * torch.pow(x - 0.3, 2)) + 3. * torch.exp(-100. * torch.pow(x - 0.7, 2)) + 0.2 * torch.exp(-50. *
                                                                                                                   torch.pow(x - 0.3, 2)) * torch.sin(x * 6.28 * 50)


def PlotTargetFunction(x_min=0, x_max=1, n=301):
  x = torch.linspace(x_min, x_max, n)

  plt.title("Target function")
  plt.xlabel("input")
  plt.ylabel("output")
  plt.plot(x, TargetFunction(x))
  plt.show()


if __name__ == "__main__":
  model = Net()
  x = torch.randn(1)
  y = TargetFunction(x)
  ypred = model.forward(x)

  print(f"x: {x.item():.3f} | y: {y.item():.3f} | ypred: {ypred.item():.3f}")

  learning_rate = 0.001
  batch_size = 64
  num_epochs = 1

  PlotTargetFunction()
