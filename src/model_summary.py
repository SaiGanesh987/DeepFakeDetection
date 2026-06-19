import tensorflow as tf

model = tf.keras.models.load_model("models/mobilenetv2.keras")

model.summary()

print("\nLayers:\n")

for i, layer in enumerate(model.layers):
    print(i, layer.name, type(layer))