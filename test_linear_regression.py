import numpy as np
from linear_regression import linear_regression

def generate_data(m, n, n_samples, x_max, x_min, offset_max, offset_min):
  x = np.random.uniform(x_min, x_max, n_samples)
  y = m * x + n
  y += np.random.uniform(x_min, x_max, n_samples)
  return x, y

def test():
    # generate data
    data_gen_dict = {"m": 7, "n": 6.1, "n_samples": 40, "x_max": 40, "x_min": -40, "offset_max": 3, "offset_min": -3}
    x, y = generate_data(**data_gen_dict)
    # test linear_regression
    linear_regression(x, y)

if __name__=="__main__":
    test()