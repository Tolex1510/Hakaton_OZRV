import torch.nn as nn
import torch
import numpy as np

class RobustAnomalyDetector(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=8, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Инициализация весов
        for layer in [self.encoder, self.decoder]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
def safe_predict(models, X):
    X_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        errors = []
        for model in models:
            outputs = model(X_tensor)
            error = torch.mean((outputs - X_tensor)**2, dim=1).numpy()
            errors.append(error)
        return np.mean(errors, axis=0)