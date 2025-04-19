import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch import nn
import pickle

class ModelTester:
    def __init__(self, model_path):
        """
        Инициализация тестера модели
        :param model_path: путь к сохраненной модели (.pth файл)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Загрузка модели и связанных параметров"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model = self._build_model(
                input_dim=checkpoint['input_dim'],
                hidden_dim=checkpoint['hidden_dim'],
                latent_dim=checkpoint['latent_dim']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.scaler = checkpoint['scaler']
            self.window_size = checkpoint['input_dim']
            self.threshold = checkpoint.get('threshold', 0.95)
            
            print(f"Модель успешно загружена. Window size: {self.window_size}")
        except Exception as e:
            raise ValueError(f"Ошибка загрузки модели: {str(e)}")
    
    def _build_model(self, input_dim, hidden_dim, latent_dim):
        """Создание архитектуры модели (должно соответствовать обучению)"""
        class AnomalyDetector(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim)
                )
            
            def forward(self, x):
                return self.decoder(self.encoder(x))
        
        return AnomalyDetector(input_dim, hidden_dim, latent_dim).to(self.device)
    
    def validate_input(self, features, target=None):
        """
        Проверка корректности входных данных
        :param features: список/массив признаков
        :param target: список/массив меток (опционально)
        :return: кортеж (нормализованные признаки, метки)
        """
        try:
            # Конвертация в numpy array
            features = np.array(features, dtype=np.float32)
            if target is not None:
                target = np.array(target, dtype=np.int32)
            
            # Проверка размеров
            if features.ndim != 1:
                raise ValueError(f"Features должен быть 1D массивом, получено {features.ndim}D")
                
            if target is not None and len(features) != len(target):
                raise ValueError(
                    f"Несоответствие размеров: features ({len(features)}), target ({len(target)})"
                )
            
            # Нормализация
            features = self.scaler.transform(features.reshape(-1, 1)).flatten()
            
            return features, target
        except Exception as e:
            raise ValueError(f"Ошибка валидации входных данных: {str(e)}")
    
    def create_sequences(self, data, labels=None):
        """Создание последовательностей с заданным window_size"""
        sequences = []
        seq_labels = []
        for i in range(len(data) - self.window_size + 1):
            sequences.append(data[i:i+self.window_size])
            if labels is not None:
                seq_labels.append(int(np.any(labels[i:i+self.window_size])))
        
        sequences = np.array(sequences)
        if labels is not None:
            return sequences, np.array(seq_labels)
        return sequences
    
    def test(self, features, target=None):
        """
        Тестирование модели на новых данных
        :param features: список/массив признаков
        :param target: список/массив меток (опционально)
        :return: словарь с результатами
        """
        try:
            # 1. Валидация входных данных
            features_norm, target_norm = self.validate_input(features, target)
            
            # 2. Создание последовательностей
            if target is not None:
                sequences, seq_labels = self.create_sequences(features_norm, target_norm)
            else:
                sequences = self.create_sequences(features_norm)
                seq_labels = None
            
            # 3. Конвертация в тензор
            sequences_tensor = torch.FloatTensor(sequences).to(self.device)
            
            # 4. Предсказание
            with torch.no_grad():
                reconstructions = self.model(sequences_tensor)
                errors = torch.mean((sequences_tensor - reconstructions)**2, dim=1).cpu().numpy()
                predictions = (errors > self.threshold).astype(int)
            
            # 5. Формирование результатов
            results = {
                'predictions': predictions,
                'errors': errors,
                'threshold': self.threshold,
                'window_size': self.window_size
            }
            
            if seq_labels is not None:
                from sklearn.metrics import classification_report
                results['classification_report'] = classification_report(
                    seq_labels, predictions, 
                    target_names=['Normal', 'Anomaly'],
                    output_dict=True
                )
            
            return results
        except Exception as e:
            raise RuntimeError(f"Ошибка тестирования: {str(e)}")

# Пример использования
if __name__ == "__main__":
    # 1. Инициализация тестера
    tester = ModelTester("Model.pth")
    
    # 2. Тестовые данные (пример)
    test_features = [1.2, 1.3, 1.2, 1.1, 15.6, 1.2, 1.3]  # Пример с аномалией
    test_target = [0, 0, 0, 0, 1, 0, 0]  # Метки (опционально)
    
    # 3. Запуск тестирования
    try:
        results = tester.test(test_features, test_target)
        
        print("\nРезультаты тестирования:")
        print(f"Обнаружено аномалий: {sum(results['predictions'])}/{len(results['predictions'])}")
        print(f"Порог ошибки: {results['threshold']:.4f}")
        
        if 'classification_report' in results:
            print("\nОтчет классификации:")
            print(f"Precision: {results['classification_report']['Anomaly']['precision']:.2f}")
            print(f"Recall: {results['classification_report']['Anomaly']['recall']:.2f}")
            print(f"F1-score: {results['classification_report']['Anomaly']['f1-score']:.2f}")
    
    except Exception as e:
        print(f"Ошибка: {str(e)}")