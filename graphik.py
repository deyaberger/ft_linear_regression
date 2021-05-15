try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as e:
    print(e)
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

class Graph():
    def __init__(self, irma):
        self.linear_line = None
        self.mse_curve = None
        self.init_linear_graph(irma)
        self.init_mse_graph()
    
    def init_linear_graph(self, irma):
        fig = plt.figure(1)
        plt.title("Linear Regression over dataset")
        plt.xlabel("Mileage (km)")
        plt.ylabel("Prices (USD)")
        kms = list(irma.dataset.kms)
        prices = list(irma.dataset.prices)
        points = plt.scatter(kms, prices, color = "pink", label="original dataset values")
        plt.pause(0.001)
        plt.waitforbuttonpress()
        plt.clf()
        plt.title("Linear Regression over dataset")
        plt.xlabel("Standardized Mileage")
        plt.ylabel("Standardized Prices")
        kms = list(irma.dataset.standardized_kms)
        prices = list(irma.dataset.standardized_prices)
        plt.scatter(kms, prices, color = "red", label="standardized dataset values")
        self.linear_x = np.linspace(irma.dataset.standardized_kms.min(), irma.dataset.standardized_kms.max())
        plt.legend()
        plt.grid(True)
        plt.pause(0.001)
        plt.waitforbuttonpress()
    
    def init_mse_graph(self):
        fig = plt.figure(2)
        plt.title("Middle squared error (MSE)")
        plt.xlabel("Episode (nb of training iterations)")
        plt.ylabel("MSE")
        self.mse_x, self.mse_y = [], []
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
        plt.legend()
        plt.pause(0.001)
        plt.waitforbuttonpress()
        
    
    def update_mse_graph(self, cost, episode):
        plt.figure(2)
        self.remove_plot(self.mse_curve)
        self.mse_x.append(episode)
        self.mse_y.append(cost)
        self.mse_curve = plt.plot(self.mse_x, self.mse_y, color="orange")
        plt.pause(0.001)
        
    
    def retrieve_original_values(self, irma):
        fig = plt.figure(1)
        fig.clf()
        plt.title("Linear Regression over dataset")
        plt.xlabel("Mileage (km)")
        plt.ylabel("Prices (USD)")
        kms = list(irma.dataset.kms)
        prices = list(irma.dataset.prices)
        points = plt.scatter(kms, prices, color = "pink", label="original dataset values")
        self.linear_x = np.linspace(0, irma.dataset.kms.max())
        self.linear_y = irma.original_theta0 + (irma.original_theta1 * self.linear_x)
        self.linear_line = plt.plot(self.linear_x, self.linear_y, color="blue", label="predict function")
        plt.legend()
        plt.grid(True)
        plt.pause(0.001)
        plt.waitforbuttonpress()
        
        
    
    def save_and_show(self, lr_name, mse_name):
        plt.figure(1)
        plt.savefig(lr_name)
        print(f"\nlinear_regression graph has been saved in '{lr_name}'")
        plt.figure(2)
        plt.savefig(mse_name)
        print(f"MSE evolution graph has been saved in '{mse_name}'")
        plt.close("all")
    