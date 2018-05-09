import numpy as np
from EasyRegression import EasyRegression
import matplotlib.pyplot as plt

data1 = {"x": np.arange(1,10), "y": np.arange(1,10)**2}
data2 = {"x": np.arange(1,6), "y": np.array([1,8,27,256,625])}
data3 = {"x": np.arange(1,10), "y": np.array([5,7,13,23,37,55,77,103,133])}

# plot data1
plt.figure()
plt.scatter(data1["x"], data1["y"])
plt.show()
# plot data2
plt.figure()
plt.scatter(data2["x"], data2["y"])
plt.show()
# plot data3
plt.figure()
plt.scatter(data3["x"], data3["y"])
plt.show()

# define models
n_epochs = 500000

# for loops dienen um weitere modelle auszuprobieren. Die aufgelisteten haben im Test am besten abgeschnotten

# data1
data1_trained_models = []
data1_models = [3]
data1_learning_rates = [1e-6]
data1_learning_rates_min = [1e-12]
for i in range(len(data1_models)):
    print("Model with {} powers".format(data1_models[i]))
    easy_reg = EasyRegression(np.zeros(data1_models[i]), plot_save_location="Aufgabe2/Data1/{}_powers".format(data1_models[i]))
    easy_reg.train(data1["x"], data1["y"], data1_learning_rates[i], n_epochs, 
                   learning_rate_min=data1_learning_rates_min[i])
    data1_trained_models.append(easy_reg)

print("Data1:")
print([model.list_of_n_powers for model in  data1_trained_models])

# data2
data2_trained_models = []
data2_models = [5]
data2_learning_rates = [1e-10]
data2_learning_rates_min = [1e-14]
for i in range(len(data2_models)):
    print("Model with {} powers".format(data2_models[i]))
    easy_reg = EasyRegression(np.zeros(data2_models[i]), plot_save_location="Aufgabe2/Data2/{}_powers".format(data2_models[i]))
    easy_reg.train(data2["x"], data2["y"], data2_learning_rates[i], n_epochs, 
                   learning_rate_min=data2_learning_rates_min[i])
    data2_trained_models.append(easy_reg)

print("Data2:")
print([model.list_of_n_powers for model in  data2_trained_models])

# data3
data3_trained_models = []
data3_models = [3]
data3_learning_rates = [1e-6]
data3_learning_rates_min = [1e-12]
for i in range(len(data3_models)):
    print("Model with {} powers".format(data3_models[i]))
    easy_reg = EasyRegression(np.zeros(data3_models[i]), plot_save_location="Aufgabe2/Data3/{}_powers".format(data2_models[i]))
    easy_reg.train(data3["x"], data3["y"], data3_learning_rates[i], n_epochs, 
                   learning_rate_min=data3_learning_rates_min[i])
    data3_trained_models.append(easy_reg)

print("Data3:")
print([model.list_of_n_powers for model in  data3_trained_models])