from event import Event

def get_deltas(parse_list: [Event]) -> [Event]:
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
            node_memory_MemFree_bytes = abs(event.node_memory_MemFree_bytes - previous.node_memory_MemFree_bytes),
            node_cpu_seconds_total = abs(event.node_cpu_seconds_total - previous.node_cpu_seconds_total),
            node_disk_read_bytes_total = abs(event.node_disk_read_bytes_total - previous.node_disk_read_bytes_total),
            node_network_receive_errs_total = abs(event.node_network_receive_errs_total - previous.node_network_receive_errs_total),
            node_processes_pids = abs(event.node_processes_pids - previous.node_processes_pids)
        ))

    return new_list