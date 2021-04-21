import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time
import random


class Datapoint():
    def __init__(self, km, price, normalized_km):
        self.km = km
        self.price = price
        self.normalized_km = normalized_km

    def __str__(self):
        return (f"km: {self.km}, price {self.price}, normalized_km = {self.normalized_km}")
        
    def __repr__(self):
        return (f"km: {self.km}, price {self.price}, normalized_km = {self.normalized_km}")


class Dataset():
    def __init__(self, data = None, path = None):
        self.initial_data = data
        self.data = data
        self.index = 0
        if (path != None):
            self.import_csv(path)

    def import_csv(self, path):
        self.data = pd.read_csv(path)
        self.data["km"] = pd.to_numeric(self.data["km"], downcast="float")
        self.data["price"] = pd.to_numeric(self.data["price"], downcast="float")
        self.kms = self.data["km"]
        self.prices = self.data["price"]
        self.normalize_km()

    def normalize(self, value, min_value, max_value):
        if value == max_value or max_value == min_value:
            return 1
        if value == min_value:
            return 0
        normalized_value = (value - min_value) / (max_value - min_value)
        return (normalized_value)

    def normalize_km(self):
        list_norm = []
        min_kms, max_kms = self.kms.min(), self.kms.max()
        for i in range(len(self.data)):
            list_norm.append(self.normalize(self.data["km"][i], min_kms, max_kms))
        self.data["normalized_km"] = list_norm

    def __getitem__(self, i):
        return Datapoint(self.data["km"][i], self.data["price"][i], self.data["normalized_km"][i])
    

    def __len__(self):
        return (self.data.shape[0])
    

    def __iter__(self):
        return (self)
    

    def __next__(self):
        if self.index < len(self.data):
            result = self[self.index]
            self.index += 1
            return(result)
        self.index = 0
        raise StopIteration
    
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
        kms = list(irma.dataset.kms)
        prices = list(irma.dataset.prices)
        plt.scatter(kms, prices, color = "red", label="dataset values")
        self.linear_x = np.linspace(0, irma.dataset.kms.max())
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
        self.mse_y.append(cost)
        self.mse_curve = plt.plot(self.mse_x, self.mse_y, color="orange", label="cost evolution")
        plt.pause(0.001)
    
class Irma():
    def __init__(self, dataset):
        self.dataset = dataset 
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.oldtheta0 = 0.0
        self.oldtheta1 = 0.0
        self.middletheta0 = 0.0
        self.middletheta1 = 0.0
        self.learning_rate = 0.01
        self.learning_rate_decay = 1.0 / 20
        self.minimal_improvement = 0.000001
        self.newcost = 0
        self.oldcost = 0
        self.middle_cost = 0


    def predict_price(self, theta0, theta1, km):
        return (theta0 + (theta1 * km))


    def error_datapoint(self, datapoint, theta0, theta1):
        error = self.predict_price(theta0, theta1, datapoint.km) - datapoint.price
        return (error)


    def squared_error_datapoint(self, datapoint, theta0, theta1):
        return ((self.error_datapoint(datapoint, theta0, theta1))**2)
    

    def mean_squared_error_dataset(self, theta0, theta1):
        total_error = 0.0
        for elem in self.dataset:
            total_error += self.squared_error_datapoint(elem, theta0, theta1)
        return (total_error / (2 * len(self.dataset)))
    
    def middle_thetas(self):
        self.middle_theta0 = (self.theta0 + self.oldtheta0) / 2
        self.middle_theta1 = (self.theta1 + self.oldtheta1) / 2

    def update_middle_error(self):
        self.middle_thetas()
        self.middle_cost = self.mean_squared_error_dataset(self.middle_theta0, self.middle_theta1)

    def update_thetas(self):
        self.oldtheta0 = self.theta0
        self.oldtheta1 = self.theta1
        sum_errors_t0 = 0.0
        sum_errors_t1 = 0.0
        for elem in self.dataset:
            sum_errors_t0 += self.error_datapoint(elem, self.theta0, self.theta1)
            sum_errors_t1 += (self.error_datapoint(elem, self.theta0, self.theta1) * elem.km) ### TODO normalized_km
        temp0 = self.theta0 - (self.learning_rate / len(self.dataset) * sum_errors_t0)
        temp1 = self.theta1 - (self.learning_rate / len(self.dataset) * sum_errors_t1)
        self.theta0 = temp0
        self.theta1 = temp1
    
    
    def should_i_keep_learning(self):
        cost_is_changing = abs(self.oldcost - self.newcost) > self.minimal_improvement
        self.update_middle_error()
        if (cost_is_changing == False):
            if (abs(self.oldcost - self.middle_cost) > self.minimal_improvement):
                if self.middle_cost < self.oldcost:
                    self.theta0, self.theta1 = self.middle_theta0, self.middle_theta1
                return True
        return cost_is_changing
    
    def decrease_and_assign(self, theta0, theta1):
        self.learning_rate = self.learning_rate * self.learning_rate_decay
        self.theta0, self.theta1 = theta0, theta1
        

    def training_loop(self, plot=False):
        go_on = True
        if plot == True:
            graphismus = Graph(irma)
            time.sleep(2)
        self.newcost = self.mean_squared_error_dataset(self.theta0, self.theta1)
        i = 0
        while (go_on):
            if plot == True:
                graphismus.update_linear_graph(self.theta0, self.theta1)
                graphismus.update_mse_graph(self.newcost, i)
                if graphismus.interactif == True:
                    plt.figure(1)
                    plt.waitforbuttonpress()
            self.oldcost = self.newcost
            self.update_thetas()
            self.newcost = self.mean_squared_error_dataset(self.theta0, self.theta1)
            go_on = self.should_i_keep_learning()
            if (self.middle_cost < self.newcost and self.middle_cost < self.oldcost):
                self.decrease_and_assign(self.middle_theta0, self.middle_theta1)
            if (self.newcost > self.oldcost):
                self.decrease_and_assign(self.oldtheta0, self.oldtheta1)
            print(self)
            i += 1
            
    def __str__(self):
        return (f"error: {self.newcost:.2E}, \tthetas: [{self.theta0}, {self.theta1}], lr: {self.learning_rate}")

    def __repr__(self):
        return (f"error: {self.newcost:.2E}, \tthetas: [{self.theta0}, {self.theta1}], lr: {self.learning_rate}")
        
if __name__ == "__main__" :
    datasetto =  Dataset(path = "data.csv")
    irma = Irma(datasetto)
    irma.training_loop(plot = True)
    # irma.test_for_graph()
    plt.show()