class Event:
    def __init__(self, time, incident, node_memory_MemFree_bytes,
                 node_cpu_seconds_total, node_disk_read_bytes_total,
                 node_network_receive_errs_total, node_processes_pids):
        self.time = time
        self.incident = incident
        self.node_memory_MemFree_bytes = node_memory_MemFree_bytes # free memory
        self.node_cpu_seconds_total = node_cpu_seconds_total # usage of CPU
        self.node_disk_read_bytes_total = node_disk_read_bytes_total # count of disk's operations
        self.node_network_receive_errs_total = node_network_receive_errs_total # errors of network
        self.node_processes_pids = node_processes_pids # count of processes
