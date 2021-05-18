import random


class Neuron:

    def __init__(self, prev_layer_size):

        self.weights = []
        self.bias = 1
        self.delta = 0
        self.value = 0

        for i in range(prev_layer_size):
            self.weights.append(random.random())

    def __str__(self):
        res = 'n' + str(self.bias)
        for i in self.weights:
            res += ',' + str(i)
        return res

    @staticmethod
    def from_file(content):
        n = Neuron(0)
        content = content.split(',')
        n.bias = float(content[0])
        for c in content[1:]:
            n.weights.append(float(c))
        return n
