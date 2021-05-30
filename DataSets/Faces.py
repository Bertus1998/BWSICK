from PIL import Image
from numpy import asarray
import glob
import FeatureExtractor
import numpy


def get_amount():
    return amount


class Face:

    def get_input(self):
        return self.landmarks

    def get_output(self):
        array = numpy.zeros(52)
        if self.status == 1:
            array[int(self.name)] = 1
        return array

    @staticmethod
    def loadImagesAndLandmarksExtract(path_to_network=None):

        face_list = []
        if path_to_network is None:
            i = 0
            for filename in glob.glob('DataBase/BD_zdjecia/Train 51/*.jpg'):
                img = Image.open(filename)
                numpydata = asarray(img)
                f = Face()
                f.landmarks, f.status = FeatureExtractor.landarmksExtract(numpydata)
                f.name = filename[len(filename) - 12:len(filename) - 9]
                face_list.append(f)


        else:
            for filename in glob.glob('DataBase/BD_zdjecia/Test 51/*.jpg'):
                img = Image.open(filename)
                numpydata = asarray(img)
                f = Face()
                f.landmarks, f.status = FeatureExtractor.landarmksExtract(numpydata)
                f.name = filename[len(filename) - 12:len(filename) - 9]
                face_list.append(f)
        return face_list
