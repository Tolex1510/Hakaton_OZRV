import pandas as pd
from Backend.Modules.event import Event
from datetime import datetime

def parse(path, attrs=None):
    """
    Parse csv file for second case and return matrix with 'time' and 'node_memory_MemFree_bytes
    :param path: Path of csv file
    :param attrs: List with attributes that you need to parse. The necessary one is 'time', and it has to be at the zero index
    :return: List with Event objects of each row in csv file with attributes: 'time' and 'node_memory_MemFree_bytes'
    """
    if not attrs:
        attrs = ["time", "incident", "node_memory_MemFree_bytes"]

    data = pd.read_csv(path)[attrs]

    result = []

    for row in data.values:
        # transfer string to datetime for attribute 'time'
        row[0] = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        result.append(row)

    return result


def delta_parse(parse_list, attrs):
    """Delta parse after all get memory_dif between next and current events in the parse_list"""
    result = []
    for i in range(len(parse_list) - 1):
        event = [parse_list[i][0]]
        for j, attr in enumerate(attrs):
            if attr in ("time", "incident"):
                event.append(parse_list[i][j])
            else:
                event.append(parse_list[i + 1][j] - parse_list[i][j])
        result.append(event)
    return result