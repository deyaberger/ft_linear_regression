import pandas as pd
import time
import matplotlib.pyplot as plt
from graphik import Graph
import numpy as np
import json


class Datapoint():
    def __init__(self, km, price, standardized_km, standardized_price):
        self.km = km
        self.price = price
        self.standardized_km = standardized_km
        self.standardized_price = standardized_price

    def __str__(self):
        return (f"km: {self.km}, price {self.price}, standardized_km = {self.standardized_km}, standardized_price = {self.standardized_price}")
        
    def __repr__(self):
        return (f"km: {self.km}, price {self.price}, standardized_km = {self.standardized_km}, standardized_price = {self.standardized_price}")
    

class Dataset():
    def __init__(self, data = None, path = None):
        self.data = data
        self.index = 0
        if (path != None):
            self.import_csv(path)
        self.get_mean_std()
        self.kms = self.data["km"]
        self.prices = self.data["price"]
        self.standardize_data()
        self.standardized_kms = self.data["standardized_kms"]
        self.standardized_prices = self.data["standardized_prices"]

    def import_csv(self, path):
        self.data = pd.read_csv(path)
        self.data["km"] = pd.to_numeric(self.data["km"], downcast="float")
        self.data["price"] = pd.to_numeric(self.data["price"], downcast="float")
    
    def get_mean_std(self):
        self.mean_kms = self.data.mean()["km"]
        self.mean_prices = self.data.mean()["price"]
        self.std_kms = self.data.std()["km"]
        self.std_prices = self.data.std()["price"]
    
    def standardize_data(self):
        standardized_kms, standardized_prices = [], []
        for ind in self.data.index:
            km, price = self.data["km"][ind], self.data["price"][ind]
            s_km = (km - self.mean_kms) / self.std_kms
            s_p = (price - self.mean_prices) / self.std_prices
            standardized_kms.append(s_km)
            standardized_prices.append(s_p)
        self.data["standardized_kms"] = standardized_kms
        self.data["standardized_prices"] = standardized_prices
        

    def __getitem__(self, i):
        return Datapoint(self.data["km"][i], self.data["price"][i], self.data["standardized_kms"][i], self.data["standardized_prices"][i])
    

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


    def predict_price(self, theta0, theta1, standardized_km):
        return (theta0 + (theta1 * standardized_km))


    def error_datapoint(self, datapoint, theta0, theta1):
        error = self.predict_price(theta0, theta1, datapoint.standardized_km) - datapoint.standardized_price
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
            sum_errors_t1 += (self.error_datapoint(elem, self.theta0, self.theta1) * elem.standardized_km) ### TODO normalized_km
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
            # time.sleep(2)
        self.newcost = self.mean_squared_error_dataset(self.theta0, self.theta1)
        print(self)
        i = 0
        while (go_on):
            if plot == True:
                graphismus.update_linear_graph(self.theta0, self.theta1)
                graphismus.update_mse_graph(self.newcost, i)
                if graphismus.interactif == True:
                    plt.figure(1) 
                    plt.waitforbuttonpress()
            tmpoldcost = self.oldcost
            self.oldcost = self.newcost
            self.update_thetas()
            self.newcost = self.mean_squared_error_dataset(self.theta0, self.theta1)
            go_on = self.should_i_keep_learning()
            if (self.middle_cost < self.newcost and self.middle_cost < self.oldcost):
                self.decrease_and_assign(self.middle_theta0, self.middle_theta1)
            self.learning_rate = self.learning_rate * 1.5
            if (self.newcost > self.oldcost):
                self.learning_rate = self.learning_rate * self.learning_rate_decay
                self.theta0, self.theta1 = self.oldtheta0, self.oldtheta1
                self.newcost = self.oldcost
                self.oldcost = tmpoldcost
            # print(self)
            i += 1

            
    def __str__(self):
        return (f"error: {self.newcost:.2E}, \tthetas: [{self.theta0}, \t{self.theta1}], \tlr: {self.learning_rate}")

    def __repr__(self):
        return (f"error: {self.newcost:.2E}, \tthetas: [{self.theta0}, \t{self.theta1}], \tlr: {self.learning_rate}")
        
if __name__ == "__main__" :
    datasetto =  Dataset(path = "data.csv")
    t = time.time()
    irma = Irma(datasetto)
    irma.training_loop(plot = False)
    # new_infos = {"t0" : irma.theta0, "t1" : irma.theta1}
    # with open("t0_t1.json", "w") as f:
    #     json.dump(new_infos, f)
    # with open("t0_t1.json", "r") as f:
    #     infos = json.load(f)
    # graphikus = Graph(irma)
    # graphikus.update_linear_graph(infos["t0"], infos["t1"])
    t1 = time.time()
    print(f"total time = {t1 - t}")
    plt.show()