import sys
sys.path.append("../") # to import EasyRegression
from EasyRegression import EasyRegression
from linear_regression import linear_regression
import numpy as np
import matplotlib.pyplot as plt

def generate_data(m, n, n_samples, x_max, x_min, offset_max, offset_min):
  x = np.random.uniform(x_min, x_max, n_samples)
  y = m * x + n
  y += np.random.uniform(x_min, x_max, n_samples)
  return x, y

def benchmark():
    # generate data 
    data_gen_dict = {"m": 7, "n": 6.1, "n_samples": 40, "x_max": 40, "x_min": -40, "offset_max": 3, "offset_min": -3}
    x, y = generate_data(**data_gen_dict)
    # linear regression
    m_lin, n_lin = linear_regression(x, y)
    lin_total_difference = np.sum(np.abs(y - (x * m_lin + n_lin)))
    # gradient descent
    easy_reg = EasyRegression(np.zeros(2))
    cost_history = easy_reg.train(x, y, 1e-3, 1001, learning_rate_min=1e-4, save_plot=True, store_cost_history=True)
    n_grad, m_grad = easy_reg.list_of_n_powers
    end_total_difference = np.sum(np.abs(y - (m_grad * x + n_grad)))
    print("Gradient descent end total_difference: " + str(end_total_difference))
    #plot 
    plt.figure()
    plt.plot([0, 1001], [lin_total_difference, lin_total_difference])
    plt.plot(np.arange(0,1001, 200), cost_history)
    plt.savefig("benchmark_lin_vs_grad")
    # print m and n
    print("Linear Regression:" + str(m_lin) + " " + str(n_lin))
    print("Gradient Descent:" + str(m_grad) + " " + str(n_grad))

benchmark()