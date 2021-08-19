from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import webbrowser
from tkinter import *


face_classifier = cv2.CascadeClassifier(
    r'D:\EMOTION BASED MUSIC PLAYER\haarcascade_frontalface_default.xml')
classifier = load_model(r'D:\EMOTION BASED MUSIC PLAYER\model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

root = Tk()
root.minsize(500, 350)

v = StringVar()
songlabel = Label(root, textvariable=v, width=50)

index = 0

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detector', frame)


    def directorychooser():
        if label == 'Happy':
            webbrowser.open('https://www.youtube.com/watch?v=25ztioI37oc')

        if label == 'Sad':
            webbrowser.open('https://www.youtube.com/watch?v=mzB1VGEGcSU&list=PL-JoUlugMMxiLdkHd537jZefZ9walk3Wp')

        if label == 'Neutral':
            webbrowser.open('https://www.youtube.com/watch?v=JYko5hDmN7Q')

        if label == 'Angry':
            webbrowser.open('https://www.youtube.com/watch?v=bywxn7unF_k')

        if label == 'Surprise':
            webbrowser.open('https://www.youtube.com/watch?v=qW-OLn3-nWE')

        if label == 'Disgust':
            webbrowser.open('https://www.youtube.com/watch?v=cI1ngpbxo7k')

        if label == 'Fear':
            webbrowser.open('https://www.youtube.com/watch?v=Veg3oK2iIFg&list=WL&index=7')

        else:
            print('NO FACE DETECTED')


    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("you are feeling", label)
        cv2.destroyWindow('Emotion Detector')
        directorychooser()
