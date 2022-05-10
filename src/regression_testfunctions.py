import torch


def TargetFunction1(x):
  return torch.exp(-20. * torch.pow(x - 0.3, 2)) + 3. * torch.exp(-100. * torch.pow(x - 0.7, 2)) + 0.2 * torch.exp(-50. *
                                                                                                                   torch.pow(x - 0.3, 2)) * torch.sin(x * 6.28 * 50)
