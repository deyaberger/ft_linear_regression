import pandas as pd

class Datapoint():
    def __init__(self, km, price):
        self.km = km
        self.price = price

    def __str__(self):
        return (f"km: {self.km}, price {self.price}")

    def __repr__(self):
        return (f"km: {self.km}, price {self.price}")


class Dataset():
    def __init__(self, data = None, path = None):
        self.data = data
        if (path != None):
            self.import_csv(path)


    def import_csv(self, path):
        self.data = pd.read_csv(path)


    def __getitem__(self, i):
        return Datapoint(self.data["km"][i], self.data["price"][i])
    

    def __len__(self):
        return (self.data.shape[0])
    
    
class Irma():
    def __init__(self, dataset):
        self.dataset = dataset 
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.oldtheta0 = 0.0
        self.oldtheta1 = 0.0
        self.learning_rate = 0.0001
        self.learning_rate_decay = 1.0 / 20
        self.minimal_improvement = 0.00001
        self.newcost = 0
        self.oldcost = 0


    def predict_price(self, km):
        return (self.theta0 + (self.theta1 * km))


    def error_datapoint(self, datapoint):
        error = self.predict_price(datapoint.km) - datapoint.price
        return (error)


    def squared_error_datapoint(self, datapoint):
        return ((self.error_datapoint(datapoint))**2)
    

    def mean_squared_error_dataset(self):
        total_error = 0.0
        for i in range(len(self.dataset)):
            total_error += self.squared_error_datapoint(self.dataset[i])
        return (total_error / (2 * len(self.dataset)))
    

    def middle_error(self):
        jesus_theta0, jesus_theta1 = self.theta0, self.theta1
        mean_theta0 = (self.theta0 + self.oldtheta0) / 2
        mean_theta1 = (self.theta1 + self.oldtheta1) / 2
        temp_cost = self.mean_squared_error_dataset()
        self.theta0, self.theta1 = jesus_theta0, jesus_theta1
        return temp_cost


    def update_thetas(self):
        # self.oldtheta0 = self.theta0
        # self.oldtheta1 = self.theta1
        sum_errors_t0 = 0.0
        sum_errors_t1 = 0.0
        for i in range(len(self.dataset)):
            sum_errors_t0 += self.error_datapoint(self.dataset[i])
            sum_errors_t1 += (self.error_datapoint(self.dataset[i]) * self.dataset[i].km)
        temp0 = self.theta0 - (self.learning_rate / len(self.dataset) * sum_errors_t0)
        temp1 = self.theta1 - (self.learning_rate / len(self.dataset) * sum_errors_t1)
        self.theta0 = temp0
        self.theta1 = temp1


    def should_i_keep_learning(self):
        cost_is_changing = abs(self.oldcost - self.newcost) > self.minimal_improvement
        if (cost_is_changing == False):
            if (abs(self.oldcost - self.middle_error()) > self.minimal_improvement):
                return True
        return cost_is_changing
        #  safety net, opti take ze middle
        
    def training_loop(self):
        go_on = True
        self.newcost = self.mean_squared_error_dataset()
        print(self)
        i = 0
        while (i < 3):
            self.oldcost = self.newcost
            self.update_thetas()
            self.newcost = self.mean_squared_error_dataset()
            go_on = self.should_i_keep_learning()
            if (self.newcost >= self.oldcost):
                self.learning_rate = self.learning_rate * self.learning_rate_decay
            print(self)
            i += 1


    def __str__(self):
        return (f"error: {self.newcost}, \tthetas: [{self.theta0}, {self.theta1}], lr: {self.learning_rate}")

    def __repr__(self):
        return (f"error: {self.newcost}, \tthetas: [{self.theta0}, {self.theta1}], lr: {self.learning_rate}")
        
        
datasetto =  Dataset(path = "data.csv")
irma = Irma(datasetto)
irma.training_loop()