import torch
from regression_model import RegressionModel
from regression_dataset import RegressionDataset


def TargetFunction(x):
  return torch.exp(-20. * torch.pow(x - 0.3, 2)) + 3. * torch.exp(-100. * torch.pow(x - 0.7, 2)) + 0.2 * torch.exp(-50. *
                                                                                                                   torch.pow(x - 0.3, 2)) * torch.sin(x * 6.28 * 50)


if __name__ == "__main__":
  train_size = 256
  train_dataset = RegressionDataset(func=TargetFunction, n=train_size)

  model = RegressionModel()
  model.train(train_dataset=train_dataset)
  model.eval(func=TargetFunction)
