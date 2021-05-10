import random


class Neuron:

    def __init__(self, prev_layer_size):

        self.weights = []
        self.bias = 1
        self.delta = 0
        self.value = 0

        for i in range(prev_layer_size):
            self.weights.append(random.random())
