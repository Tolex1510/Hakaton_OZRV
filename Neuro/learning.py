import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class AnomalyDetector(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

def calculate_memory_differences(df):
    """Вычисляет разницу MemFree между текущей и следующей строкой"""
    df['memory_diff'] = df['node_memory_MemFree_bytes'].diff().shift(-1)  # Разница со следующей строкой
    df.dropna(subset=['memory_diff'], inplace=True)  # Удаляем последнюю строку (NaN)
    return df

def create_sequences(data, window_size):
    """Создает последовательности разниц для временных рядов"""
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i+window_size])
    return np.array(sequences)

def train_model(csv_path, window_size=10, test_size=0.2, epochs=500, batch_size=32):
    # 1. Загрузка и подготовка данных
    df = pd.read_csv(csv_path)
    df = calculate_memory_differences(df)
    
    # 2. Нормализация разниц
    scaler = StandardScaler()
    diffs_normalized = scaler.fit_transform(df['memory_diff'].values.reshape(-1, 1)).flatten()[1:]
    
    # 3. Создание последовательностей
    X = create_sequences(diffs_normalized, window_size)
    
    # Если есть метки инцидентов
    if 'incident' in df.columns:
        y = np.array([1 if np.any(df['incident'].iloc[i:i+window_size]) else 0 
                     for i in range(len(df)-window_size)])
    else:
        y = np.zeros(len(X))
    
    # 4. Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
    
    # 5. Конвертация в тензоры
    train_data = torch.FloatTensor(X_train)
    train_labels = torch.FloatTensor(y_train)
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 6. Инициализация модели
    model = AnomalyDetector(input_dim=window_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 7. Обучение
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            inputs, _ = batch  # Метки не используются для автоэнкодера
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}')
    
    # 8. Сохранение модели
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'window_size': window_size
    }, 'memory_diff_model_2.pth')
    
    print("Модель успешно обучена и сохранена")
    return model, scaler

if __name__ == "__main__":
    # Конфигурация
    config = {
        "csv_path": "Hakaton_OZRV/Dataset/train.csv",
        "window_size": 10,
        "test_size": 0.2,
        "epochs": 2000,
        "batch_size": 32
    }
    
    # Запуск обучения
    train_model(**config)
