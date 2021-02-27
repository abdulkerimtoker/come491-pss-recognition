import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from .models import transformer
from .face_detector import detect_and_write_to


def get(path):
    try:
        detect_and_write_to(path, 'write.jpg')
    except:
        return None
    img = load_img('write.jpg', target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img


with tf.device('/CPU:0'):
    vgg_transformer = transformer()


def transform_x(x, y):
    with tf.device('/CPU:0'):
        x_transformed, y_transformed = vgg_transformer((x, y))
        x_transformed = x_transformed.numpy()
        y_transformed = y_transformed.numpy().astype(np.int32).reshape((len(y_transformed),))
        return x_transformed, y_transformed
