import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest
import torch
from learning import train_robust_ensemble


def main(csv_file, value_column, label_column=None, time_column=None, window_size=10, test_size=0.3):
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
    try:
        ensemble = train_robust_ensemble(X_train_tensor, window_size)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return

    # 6. Предсказание и оценка
    def safe_predict(models, X):
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            errors = []
            for model in models:
                outputs = model(X_tensor)
                error = torch.mean((outputs - X_tensor)**2, dim=1).numpy()
                errors.append(error)
            return np.mean(errors, axis=0)
    
    try:
        train_errors = safe_predict(ensemble, X_train)
        test_errors = safe_predict(ensemble, X_test)
        
        # Обучение Isolation Forest с защитой от NaN
        if np.isnan(train_errors).any():
            train_errors = np.nan_to_num(train_errors, nan=np.nanmedian(train_errors))
        
        iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        iso_forest.fit(train_errors.reshape(-1, 1))
        
        # Комбинированное предсказание
        threshold = np.quantile(train_errors, 0.95)
        iso_pred = iso_forest.predict(test_errors.reshape(-1, 1))
        y_pred = np.where((iso_pred == -1) | (test_errors > threshold), 1, 0)
        
        # Оценка качества
        if len(np.unique(y_test)) > 1:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
        else:
            print("\nOnly one class in test data. Cannot generate classification report.")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return
    
    for i, model in enumerate(ensemble):
        torch.save({
            f'model_state_dict': model.state_dict(),
            'scaler': scaler,  # Сохраняем scaler для нормализации новых данных
            'input_size': window_size,
            'feature_columns': value_column  # Сохраняем список фичей
        }, f"{i} model.pth")


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
    main(**config)