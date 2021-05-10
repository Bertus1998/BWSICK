from DataSets.Iris import *
from DataSets.Sound import *
from Network import *


def train_network(network, data_set, epochs=1000):
    for e in range(epochs):
        random.shuffle(data_set)
        print('Epoch ' + str(e))
        for d in data_set:
            network.propagate_backward(d.get_output(), network.propagate_forward(d.get_input()))


def test_network(network, data_set, threshold):
    correct = 0
    for d in data_set:
        guess = network.predict(d.get_input())
        if d.get_output()[guess] >= threshold:
            correct += 1

    print('RESULT: ' + str(correct) + ' from ' + str(len(data_set)) + ' | ' + str(int(correct * 100 / (len(data_set)))) + '%')


def network_iris():
    irises = Iris.get_irises('C:\\Users\\Adam\\Desktop\\Studia\\1 Semestr\\Biometryczne wspomaganie interakcji człowiek-komputer\\BWICK_v2\\DataBase\\Irises\\iris.data')

    network = Network([4, 5, 3])
    train_network(network, irises[0:100])
    test_network(network, irises[101:150], 1)


def network_sound():
    sounds_train = Sound.get_sounds('C:\\Users\\Adam\\Desktop\\Studia\\1 Semestr\\Biometryczne wspomaganie interakcji człowiek-komputer\\BWICK_v2\\DataBase\\BD_dzwiek\\train')
    sounds_test = Sound.get_sounds('C:\\Users\\Adam\\Desktop\\Studia\\1 Semestr\\Biometryczne wspomaganie interakcji człowiek-komputer\\BWICK_v2\\DataBase\\BD_dzwiek\\test')

    network = Network([20, 50, 2])
    train_network(network, sounds_train)
    test_network(network, sounds_test, 0.7)


# network_iris()
network_sound()
