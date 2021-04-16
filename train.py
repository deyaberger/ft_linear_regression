import pandas as pd

def calculate_cost(i):
    for i in range(self.batch_size):
        dif = hypothesis(t0, t1, data[i][0]) - data[i][1]
        temp = dif * dif
    cost = (1 / (2 * self.batch_size)) * sum(hypothesis)
    
def hypothesis(t0, t1, km):
    h = t0 + (t1 * km)
    return (h)

class Train:
    def __init__(self, data):
        self.batch_size = data.shape[0]
        self.t0 = 0
        self.t1 = 0
        self.alpha = 0.5

    
if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    