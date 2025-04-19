from sklearn.model_selection import train_test_split
import torch
from prediction import prediction
from Backend.Modules.parser import parse

def main(path):
    events = parse(path, ["time", "node_memory_MemFree_bytes"])
    time_list = [time for time, event in events]
    event_list = [event for time, event in events]

    model = torch.load("model.pth")
    X_train, X_test = train_test_split(event_list, test_size=0.2, random_state=42)
    incidents = prediction(model, X_train, X_test, task=True)

    result = [['', 'time', 'incidents']]

    for i, elem in enumerate(incidents):
        result.append([time_list[i].strftime("%Y-%m-%d %H:%M:%S"), elem])

    return result
