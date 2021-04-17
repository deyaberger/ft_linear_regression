import pandas as pd

def hypothesis(t0, t1, km):
    h = t0 + (t1 * km)
    return (h)

def squared_sum_of_diff(data, infos):
    temp = 0
    for i in range(infos.batch_size):
        dif = hypothesis(infos.t0, infos.t1, data["km"][i]) - data["price"][i]
        square = dif * dif
        temp += square
    return (temp)

def sum_of_dif_t0(data, infos):
    temp = 0
    for i in range(infos.batch_size):
        temp += hypothesis(infos.t0, infos.t1, data["km"][i]) - data["price"][i]
    return (temp)

def sum_of_dif_t1(data, infos):
    temp = 0
    for i in range(infos.batch_size):
        temp += (hypothesis(infos.t0, infos.t1, data["km"][i]) - data["price"][i]) * data["km"][i]
    return (temp)

def calculate_cost(infos, data):
    cost = (1 / (2 * infos.batch_size)) * squared_sum_of_diff(data, infos)
    return(cost)

def update_tetas(infos):
    temp0 = infos.t0 - ((infos.alpha / infos.batch_size) * sum_of_dif_t0(data, infos))
    temp1 = infos.t1 - ((infos.alpha / infos.batch_size) * sum_of_dif_t1(data, infos))
    infos.t0 = temp0
    infos.t1 = temp1

class Train:
    def __init__(self, data):
        self.batch_size = data.shape[0]
        self.t0 = 0
        self.t1 = 0
        self.alpha = 0.05
        
def find_optimal_tetas(infos, data):
    initial_cost = calculate_cost(infos, data)
    update_tetas(infos)
    new_cost = calculate_cost(infos, data)
    i = 0
    while (initial_cost >= new_cost):
        update_tetas(infos)
        i += 1
    print(f"The optimal Tetas are the following:\nt0 = {infos.t0}, t1 = {infos.t1}")
    
if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    infos = Train(data)
    find_optimal_tetas(infos, data)
    