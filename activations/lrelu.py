from .base import Activation
import numpy as np

def lrelu(x, alpha=0.01):
    return np.maximum(alpha*x, x)

def lrelu_prime(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

class LeakyReLU(Activation):    
    def __init__(self, alpha=0.01):
        super().__init__(lambda x: lrelu(x, alpha), lambda x: lrelu_prime(x, alpha))
