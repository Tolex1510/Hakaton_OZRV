import pandas as pd
from event import Event
from datetime import datetime

def parseManually(path):
    """
    Parse csv file for learning AI and first case
    :param path: Path of csv file
    :return: List with Event objects of each row in csv file with attributes: 'time', 'incident' and 'node_memory_MemFree_bytes'
    """
    data = pd.read_csv(path)[["time", "incident", "node_memory_MemFree_bytes"]]

    result = []

    # row[0] - time
    # row[1] - incident
    # row[2] - node_memory_MemFree_bytew

    for row in data.values:
        time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        result.append(Event(
            time = time,
            incident = row[1],
            node_memory_MemFree_bytes = row[2]
        ))
    
    return result

def parse(path):
    """
    Parse csv file for second case and
    :param path: Path of csv file
    :return: List with Event objects of each row in csv file with attributes: 'time' and 'node_memory_MemFree_bytes'
    """
    data = pd.read_csv(path)[["time", "node_memory_MemFree_bytes"]]

    result = []

    # row[0] - time
    # row[1] - node_memory_MemFree_bytes

    for row in data.values:
        # transfer string to datetime
        time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        result.append(Event(
            time=time,
            incident=None,
            node_memory_MemFree_bytes=row[1]
        ))

    return result