import cv2
import tensorflow as tf
import numpy as np
from zipfile import ZipFile
import os
import gdown

class PredictAge:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.conv_dict = {0: '0-3', 1: '15-20', 2: '21-27', 3: '28-34', 4: '35-43',
                          5: '4-7', 6: '44-52', 7: '53-59', 8: '60-100', 9: '8-14'}
        if not os.path.exists('model.zip'):
            try:
                gdown.download('https://drive.google.com/uc?id=1ePAkmx5izWlKBZgyhtOLx9nma-2bdj0s', 'model.zip', quiet=True)
            except:
                print('A Working Internet Connection is required to download the model (~30MB), Try Again...')
        if not os.path.exists('model'):
            with ZipFile('model.zip') as zipf:
                os.mkdir('model')
                zipf.extractall('model')
        self.model = tf.keras.models.load_model('model')
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX

    def predict_age(self, img):
        img = cv2.resize(img, (224, 224))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        age = self.model.predict(x)[0].argmax(-1)
        return self.conv_dict[age]

    def find_face(self, img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image, 1.3, 5)
        return faces

    def start_webcam_feed(self):
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            co_ords = self.find_face(frame)
            if co_ords is not ():
                for x, y, w, h in co_ords:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, self.predict_age(frame), (0, 100), self.font_face, 2, (0, 255, 0))
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()


predict_obj = PredictAge()
predict_obj.start_webcam_feed()
