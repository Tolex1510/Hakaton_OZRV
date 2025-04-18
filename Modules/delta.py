from event import Event
from typing import List

def get_deltas(parse_list: List[Event]) -> List[Event]:
    """
    Parse list with objects of Event to get delta's of data from each event
    The main idea is that if the next event more than 33% than current event, this current event is incident
    :param parse_list: List with objects of Event with data from csv file
    :return: List with objects of Event with delta's data from each event
    """
    new_list = []
    for i in range(len(parse_list) - 1):
        event = parse_list[i]
        next = parse_list[i + 1]
        dif_node_memory_MemFree_bytes = next.node_memory_MemFree_bytes - event.node_memory_MemFree_bytes
        dif_node_cpu_seconds_total = next.node_cpu_seconds_total - event.node_cpu_seconds_total
        dif_node_disk_read_bytes_total = next.node_disk_read_bytes_total - event.node_disk_read_bytes_total
        dif_node_network_receive_errs_total = next.node_network_receive_errs_total - event.node_network_receive_errs_total
        dif_node_processes_pids = next.node_processes_pids - event.node_processes_pids
        new_list.append(Event(
            time = event.time,
            incident = event.incident,
            node_memory_MemFree_bytes = f"{dif_node_memory_MemFree_bytes:.3f}",
            node_cpu_seconds_total = f"{dif_node_cpu_seconds_total:.3f}",
            node_disk_read_bytes_total = f"{dif_node_disk_read_bytes_total:.3f}",
            node_network_receive_errs_total = f"{dif_node_network_receive_errs_total:.3f}",
            node_processes_pids = f"{dif_node_processes_pids:.3f}"
        ))

    return new_list