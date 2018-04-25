import numpy as np
from EasyRegression import EasyRegression
import pandas as pd
import matplotlib.pyplot as plt

# get data
df = pd.read_csv("TabelleLinRegressionAufgabe.csv", sep=";")
x = df.Jahr.as_matrix()
y = df.Value.as_matrix()
y = np.array([float(val.replace(",", ".")) for val in y])
plt.scatter(x, y)
plt.show()

# define models
list_of_models = [2,3,4]
list_of_learning_rate = [1e-7, 1e-14, 1e-20]
list_of_learning_rate_min = [1e-9, 1e-18, 1e-25]
n_epochs = 100000

models = []

for i in range(len(list_of_models)):
    print("Model with {} powers".format(list_of_models[i]))
    easy_reg = EasyRegression(np.zeros(list_of_models[i]), plot_save_location="Aufgabe1/{}_powers".format(list_of_models[i]))
    easy_reg.train(x, y, list_of_learning_rate[i], n_epochs, 
                   learning_rate_min=list_of_learning_rate_min[i])
    models.append(easy_reg)

print([model.list_of_n_powers for model in  models])