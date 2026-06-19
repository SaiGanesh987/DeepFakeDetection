from dataset import load_datasets

train_ds, valid_ds, classes = load_datasets()

print()

print("Classes :", classes)

print()

for images, labels in train_ds.take(1):

    print("Images Shape :", images.shape)

    print("Labels Shape :", labels.shape)

    print("Image dtype :", images.dtype)

    print("Label dtype :", labels.dtype)