import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np


class Sound:

    def get_input(self):
        return self.mfcc

    def get_output(self):
        res = [0, 0]
        res[self.type] = 1
        return res

    @staticmethod
    def get_sounds(folder):

        sounds = []
        files = os.listdir(folder)
        iterator = 0

        for f in files:
            s = Sound()
            if f.startswith('F'):
                s.type = 0
            else:
                s.type = 1

            x, sr = librosa.load(folder + '\\' + f, sr=44100)

            mfcc = librosa.feature.mfcc(x, sr=sr)
            s.mfcc = []

            for i in range(len(mfcc)):
                summ = 0
                for j in mfcc[i]:
                    summ += j
                s.mfcc.append(summ / len(mfcc[i]))

            print('Extracted: ' + str(iterator))
            iterator += 1
            sounds.append(s)

        return sounds
