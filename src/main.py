from matplotlib_init import matplotlib_init
from regression_model import RegressionModel
from regression_dataset import RegressionDataset
from regression_testfunctions import TargetFunction1
from regression_testfunctions import TargetFunction2
from regression_testfunctions import TargetFunction3
from regression_testfunctions import TargetFunction4

if __name__ == "__main__":
  matplotlib_init()
  func = TargetFunction4
  train_size = 64
  test_size = 16
  train_dataset = RegressionDataset(func=func, n=train_size)
  test_dataset = RegressionDataset(func=func, n=test_size)

  model = RegressionModel()
  model.train(train_dataset=train_dataset, test_dataset=test_dataset,
              num_epochs=100, batch_size=16, learning_rate=0.002)
  model.eval(func=func, train_dataset=train_dataset)
