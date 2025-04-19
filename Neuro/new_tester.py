import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from neuro import AnomalyDetector
from sklearn.metrics import classification_report

def load_model(model_path):
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Создаем модель с конфигурацией из checkpoint
    model = AnomalyDetector(**checkpoint['model_config'])
    
    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['scaler'], checkpoint['feature_columns']

def test_model(csv_path, model_path, threshold=0.95):
    try:
        # 1. Загрузка модели и вспомогательных объектов
        model, scaler, feature_columns = load_model(model_path)
        window_size = model.encoder[0].in_features
        print(f"Model loaded. Window size: {window_size}, Features: {feature_columns}")
        
        # 2. Загрузка и предобработка данных
        df = pd.read_csv(csv_path)
        df = df[:len(df)//2]
        print(df)
        data = df[str(feature_columns)].values.astype(np.float32)

        # Обработка пропусков
        if np.isnan(data).any():
            print(f"Warning: {np.isnan(data).sum()} NaN values found. Using linear interpolation.")
            data = pd.Series(data).interpolate().values

        # Нормализация
        data = scaler.transform(data.reshape(-1, 1)).flatten()
        
        # Создание последовательностей
        sequences = np.array([data[i:i+window_size] 
                            for i in range(len(data)-window_size+1)])
        
        y_true = (sequences.sum(axis=1) > 0).astype(int)

        # 3. Предсказание
        with torch.no_grad():
            tensor_data = torch.FloatTensor(sequences)
            reconstructions = model(tensor_data)
            errors = torch.mean((tensor_data - reconstructions)**2, dim=1).numpy()
            predictions = (errors > np.quantile(errors, threshold)).astype(int)
        print(classification_report(y_true, predictions, target_names=['Normal', 'Anomaly']))
        
        # 4. Визуализация
        plt.figure(figsize=(14, 6))
        plt.plot(data, label='Data', color='blue', alpha=0.7)
        
        anomaly_indices = set()
        for i, pred in enumerate(predictions):
            if pred == 1:
                anomaly_indices.update(range(i, i+window_size))
        
        anomaly_indices = [x for x in anomaly_indices if x < len(data)]
        if anomaly_indices:
            plt.scatter(anomaly_indices, data[anomaly_indices],
                       color='red', marker='x', s=100,
                       label='Predicted Anomalies')
        
        plt.title(f'Anomaly Detection (Threshold: {threshold})')
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # 5. Статистика
        print(f"Anomalies detected: {sum(predictions)}/{len(predictions)}")
        print(f"Max error: {max(errors):.4f}")
        
        return predictions, errors
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None

# Пример использования
if __name__ == "__main__":
    test_model(
        csv_path="Dataset/train.csv",
        model_path="Model 2000.pth",
        threshold=0.95
    )