import pandas as pd
from event import Event

def parse(path):
    data = pd.read_csv(path)[["time", "incident", "node_memory_MemFree_bytes"]]

    result = []

    # row[0] - time
    # row[1] - incident
    # row[2] - node_memory_MemFree_bytew

    for row in data.values:
        result.append(Event(
            time = row[0],
            incident = row[1],
            node_memory_MemFree_bytes = row[2]
        ))
    
    return result