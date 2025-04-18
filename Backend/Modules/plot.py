import matplotlib.pyplot as plt
import parser

# Получаем список объектов Event
events = parser.parseManually('/home/bogdan/Pro/hack/train.csv') 
a = 2

# Заполняем списки
events = events[500*a-500:500*a]

x = [event.time for event in events]
y = [event.node_memory_Bounce_bytes for event in events]
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