import numpy as np
from .base import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        """
        self.weights = np.random.randn(input_size, output_size)*0.01
        self.biases = np.zeros((1, output_size))    
        """
        #Xavier/Glorot initialization
        limit=np.sqrt(6/(input_size+output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.biases = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        # compute gradients
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0)
        # update parameters
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        return input_gradient