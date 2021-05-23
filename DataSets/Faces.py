from PIL import Image
from numpy import asarray
import glob
import FeatureExtractor


class Faces:
    @staticmethod
    def loadImagesAndLandmarksExtract(path_to_network=None):
        filename_list = []
        landmarks_list = []
        i = 0
        if path_to_network is None:

            for filename in glob.glob('DataBase/BD_zdjecia/train/*.jpg'):
                print(i)
                i = i + 1
                img = Image.open(filename)
                numpydata = asarray(img)
                landmarks_list.append(FeatureExtractor.landarmksExtract(numpydata))
                filename_list.append(filename[len(filename) - 13:len(filename) - 9])
        else:
            for filename in glob.glob('DataBase/BD_zdjecia/test/*.jpg'):
                print(i)
                i = i + 1
                img = Image.open(filename)
                numpydata = asarray(img)
                landmarks_list.append(FeatureExtractor.landarmksExtract(numpydata))
                filename_list.append(filename[len(filename) - 13:len(filename) - 9])
        return landmarks_list, filename_list
