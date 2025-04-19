import matplotlib.pyplot as plt
import parser

def find_most_dense_interval(bool_list, interval_size=300):
    list_len = len(bool_list)

    if list_len == 0 or interval_size <= 0 or interval_size > list_len:
        return (None, None, 0)  # Обработка некорректных входных данных

    best_start = 0
    best_end = interval_size - 1
    max_true_count = sum(bool_list[0:interval_size]) # Считаем True в первом интервале

    # Скользящее окно:  проходим по списку, сдвигая окно
    for i in range(1, list_len - interval_size + 1):
        current_true_count = sum(bool_list[i : i + interval_size]) 
        
        if current_true_count > max_true_count:
            max_true_count = current_true_count
            best_start = i
            best_end = i + interval_size - 1

    return (best_start, best_end, max_true_count)



def plot_for_file(file_path):
    events = parser.parseManually(file_path)

    k = [bool(event.incident) for event in events]

    start_index, end_index, true_count = find_most_dense_interval(k)

    events = events[start_index:end_index+1]

    x = [event.time for event in events]
    y = [event.node_memory_MemFree_bytes for event in events]
    k = [bool(event.incident) for event in events]

    plt.plot(x, y, marker='o', label='Данные')

    # Добавление вертикальных прерывистых линий
    for i, flag in enumerate(k):
        if flag:
            plt.axvline(x[i], color='red', linestyle='--', alpha=0.7, label='Особая точка' if i == k.index(True) else "")

    # Настройки графика
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
