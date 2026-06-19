import tensorflow as tf

model = tf.keras.models.load_model("models/mobilenetv2.keras")

base_model = model.get_layer("mobilenetv2_1.00_224")

print("Base Model Name:", base_model.name)
print("=" * 60)

for layer in base_model.layers[::-1]:
    if isinstance(layer, tf.keras.layers.Conv2D):
        print("Last Conv Layer:", layer.name)
        break