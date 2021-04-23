import argparse
import pickle
import pandas
import sys
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infos', type=str, default="infos.pkl",
                        help='Enter the path and name of the pickle file where to find infos for the prediction')
    parser.add_argument("--compare", help="show comparision between actual and predicted prices",
                    action="store_true")
    parser.add_argument("--reset", help="Reset thetas to their initial values before training",
                    action="store_true")
    parser.add_argument("--round", help="Reset thetas to their initial values before training",
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

def check_info_file(args):
    if args.reset or not os.path.exists(args.infos):
        infos = {"t0" : 0, "t1" : 0, "mean_kms" : 0, "std_kms" : 0, "std_prices": 0, "mean_prices" : 0}
        if args.reset:
            with open(args.infos, "wb") as f:
                pickle.dump(infos, f)
    with open(args.infos, "rb") as f:
        infos = pickle.load(f)
    return (infos)

class Predict():
    def __init__(self, infos, km):
        self.args = args
        self.km = km
        self.theta0 = infos["t0"] if infos else 0
        self.theta1 = infos["t1"] if infos else 0
        self.mean_kms = infos["mean_kms"] if infos else 0
        self.std_kms = infos["std_kms"] if infos else 0
        self.mean_prices = infos["mean_prices"] if infos else 0
        self.std_prices = infos["std_prices"] if infos else 0
        self.s_km = None
        if self.km:
            self.standardize_km()

    def __str__(self):
        return (f"km = {self.km}, theta0 = {self.theta0}, theta1 = {self.theta1}, mean_kms = {self.mean_kms}, std_kms = {self.std_kms}, mean_prices = {self.mean_prices}, std_prices = {self.std_prices}, s_km = {self.s_km}")

    def standardize_km(self):
        self.s_km = (self.km - self.mean_kms) / self.std_kms if self.std_kms != 0 else (km - self.mean_kms)

    def unstandardized_price(self, s_p):
        price = (s_p * self.std_prices) + self.mean_prices
        return (price)

    def predict_price(self):
        s_p = self.theta0 + (self.theta1 * self.s_km)
        price = self.unstandardized_price(s_p)
        if price < 0:
            return (0)
        return (price)

    def print_comparisions(self, data_path):
        df = pandas.read_csv(data_path)
        df["predicted_prices"] = [0] * df.shape[0]
        for i in df.index:
            self.km = df["km"][i]
            self.standardize_km()
            df.at[i, "predicted_prices"] = int(self.predict_price())
        print(df[["km", 'price', 'predicted_prices']])

if __name__ == "__main__":
    args = parse_arguments()
    infos = check_info_file(args)
    if args.compare:
        km = None
    else:
        km = get_km()
    irma = Predict(infos, km)
    if args.compare:
        irma.print_comparisions("data.csv")
    else:
        if args.round == True:
            print(f"\nThe predicted price for a car that has done {km}km is: {round(irma.predict_price())}$")
        else:
            print(f"\nThe predicted price for a car that has done {km}km is: {irma.predict_price()}$")         