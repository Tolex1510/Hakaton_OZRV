import pandas as pd
from Modules.event import Event
from datetime import datetime
from typing import List
from flask import make_response
import csv
from io import StringIO

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
    Parse csv file for second case and return matrix with 'time' and 'node_memory_MemFree_bytes
    :param path: Path of csv file
    :return: List with Event objects of each row in csv file with attributes: 'time' and 'node_memory_MemFree_bytes'
    """
    data = pd.read_csv(path)[["time", "node_memory_MemFree_bytes"]]

    result = []

    for row in data.values:
        # transfer string to datetime
        elem = [datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"), row[1]]
        result.append(elem)

    return delta_parse(result)

def delta_parse(parse_list):
    result = []
    for i in range(len(parse_list) - 1):
        memory_dif = parse_list[i + 1][1] - parse_list[i + 1][1]
        result.append([parse_list[i][0], memory_dif])
    return result

def parse_csv_arr(arr):
    data = StringIO()

    csv.writer(data).writerows(arr)

    return data.getvalue()

def return_csv_response(body):
    si = StringIO()
    cw = csv.writer(si)
    cw.writerows(body)

    response = make_response(si.getvalue())
    return response