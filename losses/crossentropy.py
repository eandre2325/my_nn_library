from .base import Loss
import numpy as np

class CrossEntropy(Loss):
    
    def forward(self, y_pred, y_true):
        if not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.numpy()
        if not isinstance(y_true, np.ndarray):
            y_true = y_true.numpy()
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred))/y_true.shape[0]

    def backward(self, y_pred, y_true):
        if not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.numpy()
        if not isinstance(y_true, np.ndarray):
            y_true = y_true.numpy()

        return (-y_true / y_pred)/y_true.shape[0]