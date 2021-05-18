import os
import librosa
import numpy as np


class Sound:

    def get_input(self):
        return self.mfcc

    def get_output(self):
        return [self.type]

    @staticmethod
    def extract_feature(file_name, **kwargs):
        mfcc = kwargs.get("mfcc")
        chroma = kwargs.get("chroma")
        mel = kwargs.get("mel")
        contrast = kwargs.get("contrast")
        tonnetz = kwargs.get("tonnetz")
        X, sample_rate = librosa.core.load(file_name)
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
        return result

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

            s.mfcc = Sound.extract_feature(folder + '\\' + f, mel=True)

            print('Extracted: ' + str(iterator) + ' | Len: ' + str(len(s.mfcc)))
            iterator += 1
            sounds.append(s)

        return sounds

    @staticmethod
    def extract_and_save(folder, output_file):
        sounds = Sound.get_sounds(folder)
        file = open(output_file, mode='w')
        for s in sounds:
            file.write(str(s.type))
            for f in s.mfcc:
                file.write(',' + str(f))
            file.write('\n')
        file.close()
        return sounds

    @staticmethod
    def from_file(filepath):
        sounds = []
        file = open(filepath, mode='r')
        content = file.read()
        file.close()
        content = content.split('\n')
        for l in content[:-1]:
            data = l.split(',')
            s = Sound()
            s.type = int(data[0])
            s.mfcc = []
            for i in data[1:]:
                s.mfcc.append(float(i))
            sounds.append(s)
        return sounds


