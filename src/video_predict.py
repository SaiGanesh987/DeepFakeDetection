import cv2
import tensorflow as tf
import numpy as np
import sys

model = tf.keras.models.load_model("models/mobilenetv2.keras")

video = cv2.VideoCapture(sys.argv[1])

fake = 0
real = 0

while True:

    ret, frame = video.read()

    if not ret:
        break

    frame = cv2.resize(frame,(224,224))
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = frame.astype("float32")

    frame = np.expand_dims(frame,0)

    pred = model.predict(frame,verbose=0)[0][0]

    if pred>=0.5:
        real+=1
    else:
        fake+=1

video.release()

print()

print("Frames predicted REAL :",real)
print("Frames predicted FAKE :",fake)

print()

if real>fake:
    print("Final Prediction : REAL VIDEO")
else:
    print("Final Prediction : FAKE VIDEO")