from my_nn_library.layers.base import Layer

class Activation(Layer): # layer + activation function
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_prime(self.input)
