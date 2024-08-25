from .base import Activation
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))    

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)
