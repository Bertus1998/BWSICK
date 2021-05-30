from DataSets.Iris import *
from DataSets.Sound import *
from DataSets.Face import *
from Network import *


def train_network(network, data_set, epochs=1000):
    for e in range(epochs):
        random.shuffle(data_set)
        print('Epoch ' + str(e))
        for d in data_set:
            calculated_values = network.propagate_forward(d.get_input())
            correct_values = d.get_output()
            network.propagate_backward(correct_values, calculated_values)


def test_network(network, data_set, mode01=False):
    correct = 0
    for d in data_set:
        guess = network.predict(d.get_input())
        if mode01:
            output = d.get_output()
            print('Guess: ' + ('%.4f' % guess[1]) + '  |  Correct: ' + str(output[0]))
            if output == np.round(guess[1]):
                correct += 1
        else:
            output = d.get_output()
            print('Guess: ' + str(guess) + '  |  Correct: ' + str(output))
            if output[guess[0]] == 1:
                correct += 1

    correct_percentage = int(correct * 100 / (len(data_set)))
    print('RESULT: ' + str(correct) + ' from ' + str(len(data_set)) + ' | ' + str(correct_percentage) + '%')
    return correct_percentage


def network_iris(path_to_network=None):
    irises = Iris.get_irises(
        'C:\\Users\\Adam\\Desktop\\Studia\\1 Semestr\\Biometryczne wspomaganie interakcji człowiek-komputer\\BWICK_v2\\DataBase\\Irises\\iris.data')
    random.shuffle(irises)

    if path_to_network is None:
        network = Network([4, 5, 3])
        train_network(network, irises[0:100], epochs=1000)
    else:
        network = Network.load_from_file('Networks\\Irises\\Network_Iris_' + str(path_to_network))
    result = test_network(network, irises[100:150])
    network.save_to_file('Networks\\Irises\\Network_Iris_' + str(result))


def network_sound(path_to_network=None):
    if path_to_network is None:
        # sounds_train = Sound.extract_and_save('DataBase\\BD_dzwiek\\train', 'DataBase\\BD_dzwiek\\train_extracted')
        sounds_train = Sound.from_file('DataBase\\BD_dzwiek\\train_extracted')
        # sounds_train = Sound.get_sounds('DataBase\\BD_dzwiek\\train')
        random.shuffle(sounds_train)

        network = Network([128, 10, 1])
        train_network(network, sounds_train, epochs=100)
    else:
        network = Network.load_from_file('Networks\\Sounds\\Network_Sound_' + str(path_to_network))

    # sounds_test = Sound.extract_and_save('DataBase\\BD_dzwiek\\test', 'DataBase\\BD_dzwiek\\test_extracted')
    sounds_test = Sound.from_file('DataBase\\BD_dzwiek\\test_extracted')
    # sounds_test = Sound.get_sounds('CDataBase\\BD_dzwiek\\test')
    random.shuffle(sounds_test)

    result = test_network(network, sounds_test, mode01=True)
    network.save_to_file('Networks\\Sounds\\Network_Sound_' + str(result))


def network_faces(path_to_network=None):
    faces_train = Face.get_faces('DataBase\\BD_zdjecia\\test')

    if path_to_network is None:
        network = Network([25, 25, 25, 276])
        train_network(network, faces_train, epochs=1000)
    else:
        network = Network.load_from_file('Networks\\Faces\\Network_Faces_' + str(path_to_network))

    faces_test = Face.get_faces('DataBase\\BD_zdjecia\\train')
    result = test_network(network, faces_test)
    network.save_to_file('Networks\\Faces\\Network_Faces_' + str(result))


# network_iris()
# network_sound(100)
network_faces()
