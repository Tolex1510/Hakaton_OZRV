import pandas as pd
from event import Event
from datetime import datetime
from typing import List

ATTRIBUTES = ["time", "incident", "node_memory_MemFree_bytes", "node_cpu_seconds_total",
              "node_disk_read_bytes_total", "node_network_receive_errs_total", "node_processes_pids"]

def parseManually(path: str) -> List[Event]:
    """
    Parse csv file for learning AI and first case
    :param path: Path of csv file
    :return: List with Event objects of each row in csv file with attributes: 'time', 'incident' and 'node_memory_MemFree_bytes'
    """
    data = pd.read_csv(path)[ATTRIBUTES]

    result = []

    for row in data.values:
        time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")

        attrs = dict()
        for i, attr in enumerate(ATTRIBUTES):
            attrs[attr] = row[i]

        attrs["time"] = time
        result.append(Event(
            **attrs
        ))
    
    return result

def parse(path):
    """
    Parse csv file for second case and
    :param path: Path of csv file
    :return: List with Event objects of each row in csv file with attributes: 'time' and 'node_memory_MemFree_bytes'
    """
    data = pd.read_csv(path)[[elem for elem in ATTRIBUTES if elem != "incident"]]

    result = []

    for row in data.values:
        # transfer string to datetime
        time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")

        attrs = dict()
        for i, attr in enumerate([elem for elem in ATTRIBUTES if elem != "incident"]):
            attrs[attr] = row[i]

        attrs["time"] = time
        result.append(Event(
            **attrs
        ))

    return result