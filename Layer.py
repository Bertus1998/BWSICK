from Neuron import *


class Layer:

    def __init__(self, length, prev_layer_size):

        self.length = length
        self.neurons = []

        for i in range(length):
            self.neurons.append(Neuron(prev_layer_size))

    def __str__(self):
        res = 'l'
        for n in self.neurons:
            res += str(n)
        return res

    @staticmethod
    def from_file(content):
        l = Layer(0, 0)
        content = content.split('n')[1:]
        for c in content:
            l.neurons.append(Neuron.from_file(c))
        l.length = len(l.neurons)
        return l
