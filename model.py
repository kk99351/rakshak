import cv2
import numpy as np
import dlib
import requests
import json
import pandas as pd
from pygame import mixer
import time
import pickle
import sklearn

mixer.init()
sound1 = mixer.Sound('drowsy.wav')
sound2 = mixer.Sound('sleeping.wav')
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
filename = 'kpca_model.sav'
kpca_model = pickle.load(open(filename, 'rb'))

from imutils import face_utils

cap = cv2.VideoCapture(0)
   
   
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
sdata = {'device_id':1,
            'drowsiness_status':-1}
url = "http://192.168.113.222/nitr/api/updatedrowsinessstatus.php"

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
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
        testing_landmarks = landmarks[[36,37,38,39,40,41,42,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]]
        data = testing_landmarks.reshape(1,-1)
        data = np.concatenate((data,np.array([[1]])), axis=1)
        df_data = pd.DataFrame(data)
        testing_data = df_data.iloc[:,:-1].values
        testing_data = kpca_model.transform(testing_data)
        pred = loaded_model.predict(testing_data)

        if (pred[0] == 1):
            sleep += 1
            drowsy = 0
            active = 0
            if (sleep > 15):
                sdata['drowsiness_status'] = 1
                r=requests.post(url = url, data = sdata)
                print(r.status_code)
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                #sound2.play()
                time.sleep(1)
                #sound2.stop()
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if (active > 3):
                sdata['drowsiness_status'] = 0
                r=requests.post(url = url, data = sdata)
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