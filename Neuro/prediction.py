import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest

def prediction(model, X_train, X_test, y_test=None, task=False):
    def safe_predict(model, X):
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            errors = []
            
            outputs = model(X_tensor)
            error = torch.mean((outputs - X_tensor)**2, dim=1).numpy()
            errors.append(error)
            return np.mean(errors, axis=0)

    # try:
    print(model)
    train_errors = safe_predict(model, X_train)
    test_errors = safe_predict(model, X_test)
    
    # Обучение Isolation Forest с защитой от NaN
    if np.isnan(train_errors).any():
        train_errors = np.nan_to_num(train_errors, nan=np.nanmedian(train_errors))
    
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    iso_forest.fit(train_errors.reshape(-1, 1))
    
    # Комбинированное предсказание
    threshold = np.quantile(train_errors, 0.95)
    iso_pred = iso_forest.predict(test_errors.reshape(-1, 1))
    y_pred = np.where((iso_pred == -1) | (test_errors > threshold), 1, 0)
    
    if task:
        return y_pred
    # Оценка качества
    if len(np.unique(y_test)) > 1:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    else:
        print("\nOnly one class in test data. Cannot generate classification report.")
    # except Exception as e:
    #     print(f"Error during prediction: {str(e)}")
    #     return