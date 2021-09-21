try:
    import pandas as pd
    from graphik import Graph
    import argparse
    import pickle
except ModuleNotFoundError as e:
    print(e)
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()


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
        try:
            self.data = pd.read_csv(path)
            self.data["km"] = pd.to_numeric(self.data["km"], downcast="float")
            self.data["price"] = pd.to_numeric(self.data["price"], downcast="float")
        except Exception as e:
            print(e)
            exit()
    
    def get_mean_std(self):
        self.mean_kms = self.data.mean()["km"]
        self.mean_prices = self.data.mean()["price"]
        self.std_kms = self.data.std()["km"]
        self.std_prices = self.data.std()["price"]
    
    def standardize_data(self):
        standardized_kms, standardized_prices = [], []
        for ind in self.data.index:
            km, price = self.data["km"][ind], self.data["price"][ind]
            s_km = (km - self.mean_kms) / self.std_kms if self.std_kms != 0 else (km - self.mean_kms)
            s_p = (price - self.mean_prices) / self.std_prices if self.std_prices != 0 else (price - self.mean_prices)
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
        self.original_theta0 = self.theta0
        self.original_theta1 = self.theta1
        self.oldtheta0 = 0.0
        self.oldtheta1 = 0.0
        self.middletheta0 = 0.0
        self.middletheta1 = 0.0
        self.learning_rate = 0.01
        self.learning_rate_decay = 1.0 / 20
        self.minimal_improvement = 0.0000001
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
            sum_errors_t1 += (self.error_datapoint(elem, self.theta0, self.theta1) * elem.standardized_km)
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
        
    def retrieve_original_thetas(self, datasetto):
        self.original_theta0 = (datasetto.std_prices * self.theta0) - (((datasetto.std_prices * self.theta1) * datasetto.mean_kms) / datasetto.std_kms) + datasetto.mean_prices
        self.original_theta1 = ((datasetto.std_prices * self.theta1) / datasetto.std_kms)
        

    def training_loop(self, args):
        go_on = True
        if args.plot == True:
            graphismus = Graph(irma)
        self.newcost = self.mean_squared_error_dataset(self.theta0, self.theta1)
        print(f"\nbefore training:\t{self}")
        first_error = self.newcost
        i = 0
        while (go_on):
            if args.plot == True:
                graphismus.update_linear_graph(self.theta0, self.theta1)
                graphismus.update_mse_graph(self.newcost, i)
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
            i += 1
        print(f"\nafter training: \t{self}")
        error_diminution = (1 - (self.newcost / first_error)) * 100
        print(f"\nThanks to the training, we have decreased our prediction error of : {round(error_diminution)}%")
        self.retrieve_original_thetas(datasetto)
        print(f"\n--> And finally, after 'destandardizing' our results, theta0 = [{self.original_theta0}] and theta1 = [{self.original_theta1}]")
        if args.plot == True:
            graphismus.retrieve_original_values(self)
            graphismus.save_and_show(args.lr_name, args.mse_name)
    

    def __str__(self):
        return (f"error: {round(self.newcost, 5)}, \tthetas: [{round(self.theta0, 5)}, \t{round(self.theta1, 5)}], \tlr: {round(self.learning_rate, 5)}")

    def __repr__(self):
        return (f"error: {round(self.newcost, 5)}, \tthetas: [{round(self.theta0, 5)}, \t{round(self.theta1, 5)}], \tlr: {round(self.learning_rate, 5)}")
    
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='ft_linear_regression')
    parser.add_argument('--plot', action="store_true",
                        help='Enter "on" if you want to display the dataset with linear regression and the cost curve while training')
    parser.add_argument('--weights', type=str, default="weights.pkl",
                        help='Enter the path and name of the pickle file where to save weights at the end of the training')
    parser.add_argument('--lr_name', type=str, default="lr_graph.jpg",
                        help='Enter the path and name of the jpg file where to save the linear regression graph at the end of the training')
    parser.add_argument('--mse_name', type=str, default="mse_graph.jpg",
                        help='Enter the path and name of the jpg file where to save the MSE evolution graph at the end of the training')
    args = parser.parse_args()
    return (args)
    
def save_training_results(irma, datasetto, file_name):
    weights = {"t0" : irma.original_theta0, "t1" : irma.original_theta1}
    with open(file_name, "wb") as f:
        pickle.dump(weights, f)
    print(f"\nValues of Theta0 and Theta1 have been save in '{file_name}'\n")

    
if __name__ == "__main__" :
    args = parse_arguments()
    datasetto =  Dataset(path = "data.csv")
    irma = Irma(datasetto)
    irma.training_loop(args)
    save_training_results(irma, datasetto, args.weights)