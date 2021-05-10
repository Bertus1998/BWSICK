from Layer import *
from Neuron import *
import numpy as np


class Network:

    def __init__(self, number_of_neurons):

        self.learning_rate = .6
        self.layers = []

        for i in range(len(number_of_neurons)):
            if i == 0:
                self.layers.append(Layer(number_of_neurons[i], 0))
            else:
                self.layers.append(Layer(number_of_neurons[i], number_of_neurons[i - 1]))

    def propagate_forward(self, input_values):

        output_values = []

        # Assign inputs to input layer
        for i in range(len(self.layers[0].neurons)):
            self.layers[0].neurons[i].value = input_values[i]

        # Propagate forward
        for l in range(len(self.layers)):

            # Skip input layer
            if l == 0:
                continue

            for n in range(len(self.layers[l].neurons)):
                new_val = self.layers[l].neurons[n].bias

                for pn in range(len(self.layers[l - 1].neurons)):
                    new_val += self.layers[l].neurons[n].weights[pn] * self.layers[l - 1].neurons[pn].value

                self.layers[l].neurons[n].value = self.evaluate(new_val)

        # Set output
        for n in self.layers[-1].neurons:
            output_values.append(n.value)

        return output_values

    def propagate_backward(self, expected, recieved):

        for n in range(len(self.layers[-1].neurons)):
            error = expected[n] - recieved[n]
            self.layers[-1].neurons[n].delta = error * self.evaluate_derivative(recieved[n])

        for layer in range(len(self.layers) - 2, -1, -1):
            for n in range(len(self.layers[layer].neurons)):
                error = 0
                for nn in range(len(self.layers[layer + 1].neurons)):
                    error += self.layers[layer + 1].neurons[nn].delta * self.layers[layer + 1].neurons[nn].weights[n]

                self.layers[layer].neurons[n].delta = error * self.evaluate_derivative(self.layers[layer].neurons[n].value)

            for nn in range(len(self.layers[layer + 1].neurons)):
                for n in range(len(self.layers[layer].neurons)):
                    self.layers[layer + 1].neurons[nn].weights[n] += self.learning_rate * self.layers[layer + 1].neurons[nn].delta * self.layers[layer].neurons[n].value
                self.layers[layer + 1].neurons[nn].bias += self.learning_rate * self.layers[layer + 1].neurons[nn].delta

        error = 0
        for n in range(len(expected)):
            error += np.abs(expected[n] - recieved[n])

        error /= len(expected)
        return error

    def predict(self, input_values):
        output_values = self.propagate_forward(input_values)
        max_index = 0
        max_value = 0
        for i in range(len(output_values)):
            if output_values[i] > max_value:
                max_index = i
                max_value = output_values[i]

        return max_index

    @staticmethod
    def evaluate(value):
        return 1 / (1 + np.exp(-value))

    @staticmethod
    def evaluate_derivative(value):
        return value * (1 - value)
