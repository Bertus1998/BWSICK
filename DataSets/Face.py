from PIL import Image
import numpy as np
import os
import glob
import FeatureExtractor


class Face:
    amount = 0

    def __str__(self):
        return 'Data amount: ' + str(len(self.data)) + ' | Type: ' + str(self.type)

    def get_input(self):
        return self.data

    def get_output(self):
        result = [0] * Face.amount
        result[self.type] = 1
        return result

    @staticmethod
    def get_faces(folder):

        faces = []
        files = os.listdir(folder)
        iterator = 0
        amount_of_people = 0
        last_type = None

        for f in files:
            face = Face()

            img = Image.open(folder + '\\' + f)
            data = np.asarray(img)
            face.data = FeatureExtractor.landarmksExtract(data)

            # Skip images where extraction was impossible
            if len(face.data) == 0:
                continue

            face_type = int(f[1:4])
            if face_type != last_type:
                last_type = face_type
                amount_of_people += 1
            face.type = amount_of_people - 1

            iterator += 1
            print('Extracted: ' + str(iterator))
            faces.append(face)

        Face.amount = amount_of_people
        print('Amount of people: ' + str(Face.amount))
        return faces
