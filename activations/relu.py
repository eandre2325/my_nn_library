from .base import Activation
import numpy as np

def relu(x):
    return np.maximum(0, x)
def relu_prime(x):
    return (x > 0).astype(int)

class ReLU(Activation): 
    def __init__(self):
        super().__init__(relu, relu_prime) 