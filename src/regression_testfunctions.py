import torch


def TargetFunction1(x):
  return torch.exp(-20. * torch.pow(x - 0.3, 2)) + 3. * torch.exp(-100. * torch.pow(x - 0.7, 2))


def TargetFunction2(x):
  return torch.exp(-20. * torch.pow(x - 0.3, 2)) + 3. * torch.exp(-100. * torch.pow(x - 0.7, 2)) + 0.2 * torch.exp(-50. *
                                                                                                                   torch.pow(x - 0.3, 2)) * torch.sin(x * 6.28 * 50)


def TargetFunction3(x):
  return torch.exp(-20. * torch.pow(x - 0.3, 2)) + 3. * torch.exp(-100. * torch.pow(x - 0.7, 2)) + 0.2 * torch.exp(-50. *
                                                                                                                   torch.pow(x - 0.7, 2)) * torch.sin(x * 6.28 * 50)


def TargetFunction4(x):
  return torch.exp(-20. * torch.pow(x - 0.3, 2)) + 3. * torch.exp(-100. * torch.pow(x - 0.7, 2)) + 0.2 * torch.exp(-50. *
                                                                                                                   torch.pow(x - 0.3, 2)) * torch.sin(x * 6.28 * 200) + 0.1 * torch.exp(-50. *
                                                                                                                                                                                        torch.pow(x - 0.7, 2)) * torch.sin(x * 6.28 * 500)
