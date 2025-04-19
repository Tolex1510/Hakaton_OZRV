import torch
import torch.nn as nn
import torch.optim as optim
from neuro import RobustAnomalyDetector
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from prediction import prediction


def train_robust_ensemble(X_train, window_size, n_models=1, epochs=10, batch_size=32):
    models = []
    for i in range(n_models):
        model = RobustAnomalyDetector(input_dim=window_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        print(f"\nTraining Model {i+1}/{n_models}")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            permutation = torch.randperm(X_train.size(0))
            
            for j in range(0, X_train.size(0), batch_size):
                batch = X_train[permutation[j:j+batch_size]]
                
                # Проверка на NaN в батче
                if torch.isnan(batch).any():
                    continue
                
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (X_train.size(0) / batch_size if epoch_loss > 0 else 0)
            scheduler.step(avg_loss)
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        models.append(model)
    return models


def calculating(csv_file, value_column="node_memory_MemFree_bytes", label_column=None, time_column=None, window_size=10, test_size=0.3, learning=True):
    # 1. Загрузка и предобработка данных
    try:
        df = pd.read_csv(csv_file)
        data = df[value_column].values.astype(np.float32)
        
        # Обработка пропущенных значений
        if np.isnan(data).any():
            print(f"Warning: {np.isnan(data).sum()} NaN values found. Using linear interpolation.")
            data = pd.Series(data).interpolate().values
        
        # Нормализация всей последовательности
        scaler = StandardScaler()
        data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Создание меток (если есть)
        labels = df[label_column].values if label_column else np.zeros_like(data)
        time = df[time_column].values if time_column else np.arange(len(data))
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # 2. Создание последовательностей с проверкой
    try:
        X = np.array([data[i:i+window_size] for i in range(len(data)-window_size+1)])
        y = np.array([1 if np.any(labels[i:i+window_size]) else 0 
                     for i in range(len(labels)-window_size+1)])
        
        if len(X) == 0:
            raise ValueError("Window size too large for the data length")
    except Exception as e:
        print(f"Error creating sequences: {str(e)}")
        return

    # 3. Разделение данных с гарантией наличия обоих классов
    try:
        if len(np.unique(y)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42)
        else:
            print("Warning: Only one class in labels. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)
    except Exception as e:
        print(f"Error splitting data: {str(e)}")
        return

    # 4. Конвертация в тензоры с проверкой
    try:
        X_train_tensor = torch.FloatTensor(X_train)
        if torch.isnan(X_train_tensor).any():
            raise ValueError("NaN values detected in training data")
    except Exception as e:
        print(f"Error converting to tensors: {str(e)}")
        return

    # 5. Обучение моделей
    if learning:
        try:
            ensemble = train_robust_ensemble(X_train_tensor, window_size)
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return

    # 6. Предсказание и оценка
    task = False
    if not learning:
        task = True
        return prediction(ensemble, X_train, X_test, y_test, task)
    
    prediction(ensemble, X_train, X_test, y_test, task)

    for i, model in enumerate(ensemble):
        torch.save({
            f'model_state_dict': model.state_dict(),
            'scaler': scaler,  # Сохраняем scaler для нормализации новых данных
            'input_size': window_size,
            'feature_columns': value_column  # Сохраняем список фичей
        }, f"Model.pth")


if __name__ == "__main__":
    # Конфигурация
    config = {
        "csv_file": "Dataset/train.csv",    # Путь к CSV-файлу
        "value_column": "node_memory_MemFree_bytes",        # Столбец с данными
        "label_column": "incident",        # Опционально: столбец с метками
        "time_column": "time",     # Опционально: столбец с временем
        "window_size": 10,              # Размер окна
        "test_size": 0.3                # Доля тестовых данных
    }
    
    # Запуск
    calculating(**config)