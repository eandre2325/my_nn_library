import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class NeuralNetwork:
    def __init__(self, Layers=[], Loss=None, Learning_rate=None):
        self.layers=[]
        self.loss=None
        self.learning_rate=None
        self.grad=True

    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss=loss
    
    def set_learning_rate(self, learning_rate):
        self.learning_rate=learning_rate
    

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def print(self):
        for layer in self.layers:
            print(layer)
        print(self.loss)
        print(self.learning_rate)
    
    def train(self, x_train_loader, y_train_loader, epochs):
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            epoch_loss = 0
            for x_batch, y_batch in zip(x_train_loader, y_train_loader):
                x_batch = x_batch.numpy()
                y_batch = y_batch.numpy()
                output = self.predict(x_batch)
                loss = self.loss.forward(y_batch, output)
                epoch_loss += loss
                gradient = self.loss.backward(y_batch, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, self.learning_rate)



"""
            output = self.predict(x_train)
            loss = self.loss.forward(y_train, output)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')
            gradient = self.loss.backward(y_train, output)
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient, self.learning_rate)
"""