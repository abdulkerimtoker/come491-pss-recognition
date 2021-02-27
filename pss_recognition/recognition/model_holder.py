import threading
import tensorflow as tf
import time


class ModelHolder:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            with tf.device('/CPU:0'):
                ModelHolder._model = tf.keras.models.load_model('models/latest')
        return cls._model


def load_trained_model():
    while True:
        time.sleep(30)
        with tf.device('/CPU:0'):
            ModelHolder._model = tf.keras.models.load_model('models/latest')


model_loader_thread = threading.Thread(target=load_trained_model)
model_loader_thread.start()
