import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def make_plot(m, n, x, y):
    """this function makes a plot of the function predicted and the points given
    
    Arguments:
        m {float} -- The m of the function
        n {float} -- The n of the function
        x {np.array} -- The x coordinates of the data
        y {np.array} -- The y coordinates of the data
    """
    plt.scatter(x,y)
    pred_x = np.arange(x.min()-10, x.max()+10)
    pred_y = m*pred_x+n
    plt.plot(pred_x, pred_y)
    plt.show()


def linear_regression(x,y):
    """This is the function to do normal linear regression
    It provides a table and a plot where you can see how accurate the function is
    
    Arguments:
        x {np.array} -- the x coordinates of the data
        y {np.array} -- the y coordinates of the data
    
    Returns:
        float -- the m of the function
        float -- the y of the function
    """

    
    x_mid = np.mean(x)
    y_mid = np.mean(y)
    data = {"x_data": x, "y_data": y, "x": x-x_mid, "y": y-y_mid, "xy": (x-x_mid)*(y-y_mid),
            "xx": (x-x_mid)**2, "yy": (y-y_mid)**2}
    df = pd.DataFrame(data=data)
    sums = {name: np.sum(x) for name, x in data.items()}
    df.append(sums, ignore_index=True)
    print(df.head())
    m = sums["xy"] / sums["xx"]
    n = y_mid - x_mid * m
    make_plot(m, n, x, y)
    difference = np.sum([np.abs(y[i]-(m*x[i]+n)) for i in range(len(x))])
    print("Total difference = " + str(difference))
    print("Avg Distance = " + str(difference / len(x)))
    return m, n
