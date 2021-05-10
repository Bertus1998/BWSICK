from Neuron import *


class Layer:

    def __init__(self, length, prev_layer_size):

        self.length = length
        self.neurons = []

        for i in range(length):
            self.neurons.append(Neuron(prev_layer_size))
