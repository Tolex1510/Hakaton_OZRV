import pandas as pd

def converter(path):
    data = pd.read_csv(path)[["time", "incident"]]

    data.to_csv(path)

if __name__ == "__main__":
    converter("../Uploads/train.csv")