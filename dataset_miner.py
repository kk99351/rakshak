import cv2
import numpy as np
import dlib
import pandas as pd
from pygame import mixer

mixer.init()
sound1 = mixer.Sound('drowsy.wav')
sound2 = mixer.Sound('sleeping.wav')

from imutils import face_utils
   
   
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

vid_count = 43
color = (0, 0, 0)

while (vid_count):
    cap = cv2.VideoCapture('vid ('+str(vid_count)+').mp4')
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
        landmarks = landmarks[[36,37,38,39,40,41,42,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]]
        print(landmarks)
        data = landmarks.reshape(1,-1)
        data = np.concatenate((data,np.array([[0]])), axis=1)
        df_data = pd.read_csv('file1.csv').values
        df_data = np.delete(df_data, 0, axis=1)
        data = np.concatenate((data, df_data), axis=0)
        df_data = pd.DataFrame(data)
        df_data.to_csv('file1.csv')
    vid_count = vid_count -1
    key = cv2.waitKey(1)
    if key == 27:
        break