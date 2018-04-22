import numpy as np
from EasyRegression import EasyRegression

def generate_data(list_of_n_powers, n_samples, x_min, x_max, offset_min, offset_max):
  x = np.random.uniform(x_min, x_max, n_samples)
  y = np.sum([list_of_n_powers[i]*x**i for i in range(len(list_of_n_powers))], axis=0)
  y += np.random.uniform(offset_min, offset_max, n_samples)
  return x, y

def test():
    # generate data
    data_gen_dict = {"list_of_n_powers" : [2,3, 0.5, 0.4, 0.02], "n_samples" : 20, "x_min" : -20, "x_max" : 20, "offset_min" : -20, "offset_max" : 20}
    x, y = generate_data(**data_gen_dict)
    # test Easyregression
    easy_reg = EasyRegression(np.zeros(len(data_gen_dict["list_of_n_powers"])+4))
    easy_reg.train(x, y, 1e-21, 1001000, learning_rate_min=1e-50)

if __name__=="__main__":
    test()
