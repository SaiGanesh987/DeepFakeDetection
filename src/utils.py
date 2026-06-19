import cv2
from pathlib import Path


def create_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_frame(frame, save_path, image_size):
    frame = cv2.resize(frame, image_size)
    cv2.imwrite(str(save_path), frame)