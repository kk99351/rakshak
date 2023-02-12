import cv2
import numpy as np
import dlib
import requests
import json
import pandas as pd
from pygame import mixer
import time

mixer.init()
sound1 = mixer.Sound('drowsy.wav')
sound2 = mixer.Sound('sleeping.wav')

from imutils import face_utils

cap = cv2.VideoCapture(0)
   
   
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
data = {'device_id':1,
            'drowsiness_status':-1}
url = "http://192.168.113.222/nitr/api/updatedrowsinessstatus.php"

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking if it is blinked
    if ratio > 0.25:
        return 2
    elif (ratio > 0.2) and (ratio <= 0.25):
        return 1
    else:
        return 0

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])

    if distance > 18:
        return 1
    else:
        return 2


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    # detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            # Setting the radius of circles=2 pixels, colur=white
            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        yawning = lip_distance(landmarks)

        if (left_blink == 0 or right_blink == 0 or yawning == 0 or yawning == 0):
            sleep += 2
            active = 0
            if (sleep > 15):
                data['drowsiness_status'] = 1
                r=requests.post(url = url, data = data)
                print(r.status_code)
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                #sound2.play()
                time.sleep(1)
                #sound2.stop()
        elif (left_blink == 1) or (right_blink == 1) or (yawning == 1) or (yawning == 1):
            sleep += 1
            active = 0
        else:
            sleep = 0
            active += 1
            if (active > 6):
                data['drowsiness_status'] = 0
                r=requests.post(url = url, data = data)
                print(r.status_code)
                status = "Active"
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", frame.copy())
    key = cv2.waitKey(1)
    if key == 27:
        data['drowsiness_status'] = 2
        r=requests.post(url = url, data = data)
        print(r.status_code)
        break