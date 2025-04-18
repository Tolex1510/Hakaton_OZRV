from event import Event
from typing import List

def get_deltas(parse_list: List[Event]) -> List[Event]:
    """
    Parse list with objects of Event to get delta's of data from each event
    :param parse_list: List with objects of Event with data from csv file
    :return: List with objects of Event with delta's data from each event
    """
    new_list = []
    for i in range(1, len(parse_list)):
        event = parse_list[i]
        previous = parse_list[i - 1]
        new_list.append(Event(
            time = event.time,
            incident = event.incident,
            node_memory_MemFree_bytes = f"{(event.node_memory_MemFree_bytes - previous.node_memory_MemFree_bytes)/previous.node_memory_MemFree_bytes:.3f}",
            node_cpu_seconds_total = f"{(event.node_cpu_seconds_total - previous.node_cpu_seconds_total)/previous.node_cpu_seconds_total:.3f}",
            node_disk_read_bytes_total = f"{(event.node_disk_read_bytes_total - previous.node_disk_read_bytes_total)/previous.node_disk_read_bytes_total:.3f}",
            node_network_receive_errs_total = f"{(event.node_network_receive_errs_total - previous.node_network_receive_errs_total)/previous.node_network_receive_errs_total:.3f}",
            node_processes_pids = f"{(event.node_processes_pids - previous.node_processes_pids)/previous.node_processes_pids:.3f}"
        ))

    return new_list