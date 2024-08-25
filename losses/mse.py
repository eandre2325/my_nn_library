from .base import Loss
import numpy as np

class MSE(Loss):
    def forward(self, y_pred, y_true):
        if not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.numpy()
        if not isinstance(y_true, np.ndarray):
            y_true = y_true.numpy()
        return np.mean((y_pred - y_true)**2)
    
    def backward(self, y_pred, y_true):
        return 2*(y_pred - y_true) / y_true.size
    