from regression_model import RegressionModel
from regression_dataset import RegressionDataset
from regression_testfunctions import TargetFunction1
from matplotlib_init import matplotlib_init

if __name__ == "__main__":
  matplotlib_init()
  train_size = 64
  test_size = 16
  train_dataset = RegressionDataset(func=TargetFunction1, n=train_size)
  test_dataset = RegressionDataset(func=TargetFunction1, n=test_size)

  model = RegressionModel()
  model.train(train_dataset=train_dataset, test_dataset=test_dataset,
              num_epochs=200, batch_size=16, learning_rate=0.002)
  model.eval(func=TargetFunction1, train_dataset=train_dataset)
