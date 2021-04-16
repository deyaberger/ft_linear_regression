import json

def error_msg(msg):
    print(msg)
    return (False)

def check_input(arg):
    try:
        km = float(arg)
        if km < 0:
            return (error_msg("Please enter a number equal or greater than 0"))
        if km > 1000000:
            return (error_msg("Please enter a number smaller than a million, a car cannot survive after that many kilometers..."))
        return (True)
    except:
        return (error_msg("Error: Please enter a string that can be converted to a integer or a float"))

def get_km():
    print("Please write the number of kilometers of the car so we can predict its price, then press ENTER:")
    arg = input()
    while not check_input(arg):
        arg = input()
    km = float(arg)
    return (km)

def predict_price(km, t0, t1):
    print(f"{t0} + ({t1} * {km})")
    p = t0 + (t1 * km)
    return (p)

if __name__ == "__main__":
    km = get_km()
    with open("t0_t1.json", "r") as f:
        tetas = json.load(f)
    t0, t1 = tetas["t0"], tetas["t1"]
    print(f"\nThe predicted price for a car that has done {km}km is: {predict_price(km, t0, t1)}$")
    