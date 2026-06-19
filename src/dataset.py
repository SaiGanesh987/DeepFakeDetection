import tensorflow as tf
from config import *

AUTOTUNE = tf.data.AUTOTUNE


def load_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True,
        seed=42
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )

    train_ds = train_ds.prefetch(AUTOTUNE)

    valid_ds = valid_ds.prefetch(AUTOTUNE)

    return train_ds, valid_ds