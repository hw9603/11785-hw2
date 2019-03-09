from layers import *

class CNN_B():
    def __init__(self):
        # Your initialization code goes here
        self.layers = [Conv1D(24, 8, 8, 4), ReLU(), Conv1D(8, 16, 1, 1), ReLU(), Conv1D(16, 4, 1, 1), Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        w = [(8, 8, 24), (16, 1, 8), (4, 1, 16)]
        for i, weight in enumerate(weights):
            self.layers[i * 2].W = weight.T.reshape(w[i]).transpose(0, 2, 1)

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta




class CNN_C():
    def __init__(self):
        # Your initialization code goes here
        self.layers = [Conv1D(24, 2, 2, 2), ReLU(), Conv1D(2, 8, 2, 2), ReLU(), Conv1D(8, 4, 2, 1), Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        w = [(2, 2, 24), (8, 2, 2), (4, 2, 8)]
        for i, weight in enumerate(weights):
            out_channel, kernel, in_channel = w[i]
            weight = weight[:in_channel * kernel, :out_channel]
            self.layers[i * 2].W = weight.T.reshape(w[i]).transpose(0, 2, 1)

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
