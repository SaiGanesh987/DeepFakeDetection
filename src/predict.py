import tensorflow as tf
import numpy as np
import cv2
import sys

MODEL_PATH = "models/mobilenetv2.keras"

model = tf.keras.models.load_model(MODEL_PATH)

image_path = sys.argv[1]

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.resize(image, (224,224))

image = image.astype("float32")

image = np.expand_dims(image, axis=0)

prediction = model.predict(image, verbose=0)[0][0]

if prediction >= 0.5:
    print(f"\nPrediction : REAL")
    print(f"Confidence : {prediction*100:.2f}%")
else:
    print(f"\nPrediction : FAKE")
    print(f"Confidence : {(1-prediction)*100:.2f}%")