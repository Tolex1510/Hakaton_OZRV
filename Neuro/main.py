from sklearn.model_selection import train_test_split
import torch
from prediction import prediction

def main(events):
    time_list = [time for time, event in events]
    event_list = [event for time, event in events]

    model = torch.load("model.pth")
    X_train, X_test = train_test_split(event_list, test_size=0.2, random_state=42)
    result = prediction(model, X_train, X_test, task=True)

    return result
