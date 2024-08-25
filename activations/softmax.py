from .base import Activation
import numpy as np

def softmax(x):
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

def softmax_prime(x):
    return softmax(x) * (1 - softmax(x))

class Softmax(Activation):
    def __init__(self):
        super().__init__(softmax, softmax_prime)

