try:
    import argparse
    import pickle
    import pandas
    import sys
    import os
except ModuleNotFoundError as e:
    print(e)
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="weights.pkl",
                        help='Enter the path and name of the pickle file where to find weights for the prediction')
    parser.add_argument("--compare", help="show comparision between actual and predicted prices",
                    action="store_true")
    parser.add_argument("--reset", help="Reset thetas to their initial values before training",
                    action="store_true")
    args = parser.parse_args()
    return (args)

def error_msg(msg):
    print(msg)
    return (False)

def check_input(arg):
    try:
        km = float(arg)
        if km <= 0:
            return (error_msg("\nPlease enter a number greater than 0"))
        if km > sys.maxsize:
            return (error_msg("\nHum sorry but batman's car doesn't exist, humans car usually don't exceed 190 000km, please enter a realistic number of km"))
        return (True)
    except:
        return (error_msg("\nError: Please enter a single string that can be converted to an integer or a float"))

def get_km():
    print("Please write the number of kilometers of the car so we can predict its price, then press ENTER:")
    arg = input()
    while not check_input(arg):
        arg = input()
    km = float(arg)
    return (km)

def check_weights_file(args):
    weights = {"t0" : 0, "t1" : 0}
    if args.reset:
        with open(args.weights, "wb") as f:
            pickle.dump(weights, f)
        print(f"--> Reseting the values of thetas to 0, saving it in the file '{args.weights}'")
    if os.path.exists(args.weights):
        with open(args.weights, "rb") as f:
            weights = pickle.load(f)
            if not args.reset:
                print(f"--> Loading the values of thetas from the file '{args.weights}'")
    print(f"theta0 = [{round(weights['t0'], 3)}]\ttheta1 = [{round(weights['t1'], 3)}]\n")
    return (weights)

class Predict():
    def __init__(self, weights, km):
        self.args = args
        self.km = km
        self.theta0 = weights["t0"] if weights else 0
        self.theta1 = weights["t1"] if weights else 0

    def predict_price(self):
        price = self.theta0 + (self.theta1 * self.km)
        return (price)

    def print_comparisions(self, data_path):
        df = pandas.read_csv(data_path)
        df["predicted_prices"] = [0] * df.shape[0]
        for i in df.index:
            self.km = df["km"][i]
            df.at[i, "predicted_prices"] = int(self.predict_price())
        print(f"--- Actual Prices from dataset Vs predicted prices with our linear regression: ---\n")
        print(df[["km", 'price', 'predicted_prices']].to_string(index = False))

if __name__ == "__main__":
    args = parse_arguments()
    weights = check_weights_file(args)
    if args.compare:
        km = None
    else:
        km = get_km()
    irma = Predict(weights, km)
    if args.compare:
        irma.print_comparisions("data.csv")
    else:
        print(f"\n--> The predicted price for a car that has done {km}km is: {irma.predict_price()}$")         