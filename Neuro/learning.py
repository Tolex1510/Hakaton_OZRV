import torch
import torch.nn as nn
import torch.optim as optim
from neuro import RobustAnomalyDetector

model_path = "binary_classifier.pth"

def train_robust_ensemble(X_train, window_size, n_models=1, epochs=50, batch_size=32):
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

