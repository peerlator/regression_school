import numpy as np
import matplotlib.pyplot as plt

class EasyRegression:
    def __init__(self, list_of_n_powers, plot_save_location="easy_reg_plots/pred_func"):
        """ This is the init function of EasyRegression
        Arguments:
            list_of_n_powers {list} - the starting list of of powers
        Keyword Arguments:
            plot_save_location {string} - the location to save the plots
        """
        
        self.list_of_n_powers = list_of_n_powers
        self.plot_save_loacation = plot_save_location

    def compute_output(self, x):
        """Computes the Output of the list_of_powers, with input x
        Arguments:
            x {np.array} - The x data
        """
        y = np.sum([self.list_of_n_powers[i]*x**i for i in range(len(self.list_of_n_powers))], axis=0)
        return y

    def train(self, x, y, learning_rate, n_epochs, learning_rate_decay=True, learning_rate_min=None, save_plot=True, store_cost_history=False):
        """Trains based on Gradient descent
        Arguments:
            x: The x data to train on
            y: The y data to train on
            learning_rate: Learning Rate describes how fast the model learns
            n_epochs: The number of epochs the model should train.
        Keyword Arguments:
            learning_rate_decay: learning rate decay says, that the learning rate decreases over time
            learning_rate_min: Where the learning rate ends
        """
        this_learning_rate_min = learning_rate_min # needed to do this to not overwrite standard value
        if not learning_rate_min:
            this_learning_rate_min = learning_rate / 10
        learning_rate_decay_step = (learning_rate - this_learning_rate_min) / n_epochs
        cost_history = []
        for i in range(n_epochs):
            cost, avg_cost = self._next_epoch(x, y, learning_rate)
            if i % 200 == 0:
                if store_cost_history:
                    cost_history.append(cost)
                print("Epoch {}, Cost: {}, Avg_Cost: {}".format(i, cost, avg_cost))
            if i % 10000 == 0 and save_plot:
                plt.figure()
                x_plot = np.arange(x.min()-10, x.max()+10)
                y_plot = self.compute_output(x_plot)
                plt.scatter(x, y) # plot data
                plt.plot(x_plot, y_plot) # plot predicted_function
                plt.savefig(self.plot_save_loacation + str(i))
        if store_cost_history:
            return cost_history

    def _next_epoch(self, x, y, learning_rate):
        """ The next epoch (internal_function)
        Arguments:
            x: The x training data
            y: The y training data
            learning_rate: The amount of change made
        """
        preds = self.compute_output(x)
        error = self._compute_error(preds, y)
        cost = np.sum(np.abs(error))
        avg_cost = cost / len(x)
        new_list_of_n_powers = []
        for i in range(len(self.list_of_n_powers)):
            this_gradient = 2 / len(x) * np.sum(-(x**i) * error)
            new_list_of_n_powers.append(self.list_of_n_powers[i] - this_gradient * learning_rate)
        
        self.list_of_n_powers = new_list_of_n_powers
        return cost, avg_cost

    def _compute_error(self, preds, y):
        """Computes the error of preds and y
        Arguments:
            preds: The predicted value, based on x of y
            y: The true data 
         """
        return y - preds
