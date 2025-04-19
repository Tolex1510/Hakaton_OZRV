from sklearn.model_selection import train_test_split
import torch
from prediction import prediction
import pandas as pd
from neuro import AnomalyDetector
from sklearn.preprocessing import StandardScaler
import numpy as np

def calculate_memory_differences(df):
    """Вычисляет разницу MemFree между текущей и следующей строкой"""
    df['memory_diff'] = df['node_memory_MemFree_bytes'].diff().shift(-1)
    df.dropna(subset=['memory_diff'], inplace=True)
    return df

def create_sequences(data, window_size):
    """Создает последовательности разниц для временных рядов"""
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i+window_size])
    return np.array(sequences)

def prepare_data(df, window_size):
    """Подготавливает данные для модели"""
    # Вычисляем разницы
    df = calculate_memory_differences(df)
    
    # Нормализуем разницы
    scaler = StandardScaler()
    diffs_normalized = scaler.fit_transform(df['memory_diff'].values.reshape(-1, 1)).flatten()[1:]
    
    # Создаем последовательности
    X = create_sequences(diffs_normalized, window_size)
    
    # Создаем метки (если есть столбец 'incident')
    if 'incident' in df.columns:
        y = np.array([1 if np.any(df['incident'].iloc[i:i+window_size]) else 0 
                     for i in range(len(df)-window_size)])
    else:
        y = np.zeros(len(X))
    
    return X, y, scaler

def main(file="Dataset/train.csv", model_path="Model 5000.pth", window_size=10):
    # 1. Загрузка данных
    events = pd.read_csv(file)
    
    # 2. Подготовка данных
    X, y, scaler = prepare_data(events, window_size)
    
    # 3. Загрузка модели
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = AnomalyDetector(input_dim=checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 4. Разделение данных (на числовых массивах, а не DataFrame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # 5. Предсказание (передаем numpy массивы, а не DataFrame)
    result = prediction(model, X_train, X_test, y_test, task=True)
    print(result)
    return result

if __name__ == "__main__":
    main()