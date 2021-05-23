import cv2
import sys
import dlib
from math import sqrt
import imageio
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def faceDetection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    # Draw a rectangle around the faces
    for face in faces:
        landmarks = predictor(gray, face)
        print(landmarks)
        i = 0
        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            print(x, y)
            cv2.circle(frame, (x, y), 3, (255, 0, 0))

    return frame


def extractionFeatureOfFace(frame, face):
    landmarks = predictor(frame, face)
    print(landmarks.shape)
    print('\n\nFirst 10 landmarks:\n', landmarks[:10])


def landarmksExtract(frame):
    landmarks_list = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        crop_img = frame[y-5:y + h+5, x-5:x + w+5]
        dim = (120, 120)
        resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
        resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        tada = detector(resized_gray)
        x = 0
        if len(tada) != 0:
            landmarks = predictor(resized_gray, tada[0])
            # 37 i 40
            landmarks_list.append(length(landmarks.part(36), landmarks.part(39)))
            # 38 i 41
            landmarks_list.append(length(landmarks.part(37), landmarks.part(40)))
            # 39 i 42
            landmarks_list.append(length(landmarks.part(38), landmarks.part(41)))
            # 37 i 40 half
            landmarks_list.append(length(landmarks.part(36), landmarks.part(39)))
            # 43 i 46
            landmarks_list.append(length(landmarks.part(42), landmarks.part(45)) / 2)
            # 18 i 22
            landmarks_list.append(length(landmarks.part(17), landmarks.part(21)))
            # 23 i 27
            landmarks_list.append(length(landmarks.part(22), landmarks.part(26)))
            landmarks_list.append(length(landmarks.part(21), landmarks.part(22)))
            landmarks_list.append(length(landmarks.part(27), landmarks.part(33)))
            landmarks_list.append(length(landmarks.part(31), landmarks.part(35)))
            landmarks_list.append(length(landmarks.part(48), landmarks.part(54)))
            landmarks_list.append(length(landmarks.part(51), landmarks.part(57)))
            landmarks_list.append(length(landmarks.part(33), landmarks.part(8)))
            landmarks_list.append(length(landmarks.part(0), landmarks.part(36)))
            landmarks_list.append(length(landmarks.part(16), landmarks.part(45)))
            landmarks_list.append(length(landmarks.part(19), landmarks.part(8)))
            landmarks_list.append(length(landmarks.part(24), landmarks.part(8)))
            landmarks_list.append(length(landmarks.part(51), landmarks.part(62)))
            landmarks_list.append(length(landmarks.part(66), landmarks.part(57)))
            landmarks_list.append(length(landmarks.part(48), landmarks.part(61)))
            landmarks_list.append(length(landmarks.part(64), landmarks.part(54)))
            landmarks_list.append(length(landmarks.part(39), landmarks.part(27)))
            landmarks_list.append(length(landmarks.part(27), landmarks.part(42)))

            # proporcje
            landmarks_list.append(
                length(landmarks.part(27), landmarks.part(33)) / length(landmarks.part(31), landmarks.part(35)))
            landmarks_list.append(
                length(landmarks.part(51), landmarks.part(62)) / length(landmarks.part(66), landmarks.part(57)))
        else:
            x = x + 1
            print(x)

    return landmarks_list


def length(p1, p2):
    return round(sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2), 3)
