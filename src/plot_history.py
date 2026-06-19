import pickle
import matplotlib.pyplot as plt

with open("history.pkl", "rb") as f:
    history = pickle.load(f)

# Accuracy
plt.figure(figsize=(8,5))
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("outputs/accuracy_curve.png")
plt.show()

# Loss
plt.figure(figsize=(8,5))
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("outputs/loss_curve.png")
plt.show()