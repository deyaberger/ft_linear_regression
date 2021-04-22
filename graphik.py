import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time
from math import log

class Graph():
    def __init__(self, irma, interactif=False):
        self.interactif = interactif
        self.linear_line = None
        self.mse_curve = None
        self.init_linear_graph(irma)
        self.init_mse_graph()
    
    def init_linear_graph(self, irma):
        fig = plt.figure(1)
        plt.title("Linear Regression")
        plt.xlabel("Mileage (km)")
        plt.ylabel("Prices (USD)")
        kms = list(irma.dataset.standardized_kms)
        prices = list(irma.dataset.standardized_prices)
        plt.scatter(kms, prices, color = "red", label="dataset values")
        self.linear_x = np.linspace(irma.dataset.standardized_kms.min(), irma.dataset.standardized_kms.max())
        plt.legend()
        plt.grid(True)
        fig.canvas.manager.window.attributes("-topmost", 1)
        if not self.interactif:
            plt.pause(0.001)
    
    def init_mse_graph(self):
        fig = plt.figure(2)
        plt.title("MSE cost evolution")
        plt.xlabel("Episode")
        plt.ylabel("Cost")
        self.mse_x, self.mse_y = [], []
        fig.canvas.manager.window.attributes("-topmost", 0)
        plt.pause(0.001)
    
    def remove_plot(self, a_plot):
        if a_plot:
            line = a_plot.pop(0)
            line.remove()
            
    def update_linear_graph(self, theta0, theta1):
        plt.figure(1)
        self.remove_plot(self.linear_line)
        self.linear_y = theta0 + (theta1 * self.linear_x)
        self.linear_line = plt.plot(self.linear_x, self.linear_y, color="blue", label="predict function")
        if not self.interactif:
            plt.pause(0.001)
    
    def update_mse_graph(self, cost, episode):
        plt.figure(2)
        self.remove_plot(self.mse_curve)
        self.mse_x.append(episode)
        self.mse_y.append(log(cost, 10))
        self.mse_curve = plt.plot(self.mse_x, self.mse_y, color="orange", label="cost evolution")
        plt.pause(0.001)
    