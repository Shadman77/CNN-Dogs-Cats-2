import cv2
import tensorflow as tf

CATEGORIES = ["Dog", "Cat"]

model = tf.keras.models.load_model("../training/64x3-A-CNN.model")

def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)