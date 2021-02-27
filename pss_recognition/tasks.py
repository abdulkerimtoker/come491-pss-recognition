from pss_recognition import celery_app as app

import json
import time


@app.task
def sea():
    print("sea")


@app.task
def train():
    from pss_recognition.models.picture import Picture
    from django.conf import settings
    from .recognition.models import my_model
    from .recognition.recognition import get, vgg_transformer

    import numpy as np
    import tensorflow as tf

    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass

    pictures = Picture.objects.all()
    person_ids = Picture.objects.values('person_id').distinct()
    name_to_label = json.loads(open('data/name_to_label.json', 'r').read())
    registered_labels = []
    for row in person_ids:
        name_to_label[str(row['person_id'])] = len(name_to_label)
        registered_labels.append(name_to_label[str(row['person_id'])])
    label_to_name = {v: k for k, v in name_to_label.items()}

    x = []
    y = []

    for picture in pictures:
        img = get(settings.PICTURES_PATH + str(picture.id))
        if img is not None:
            x.append(img)
            y.append(name_to_label[str(picture.person_id)])

    x = np.asarray(x)
    y = np.asarray(y)

    with tf.device('/CPU:0'):
        x_transformed, y_transformed = vgg_transformer.predict((x, y))

    y_transformed = y_transformed.reshape((len(y),)).astype(np.int32)

    x_pre_transformed = np.load('data/x_transformed.npy')
    y_pre_transformed = np.load('data/y_transformed.npy')
    x_test_pre_transformed = np.load('data/x_test_transformed.npy')
    y_test_pre_transformed = np.load('data/y_test_transformed.npy')

    x = np.concatenate((x_transformed, x_pre_transformed))
    y = np.concatenate((y_transformed, y_pre_transformed))

    x_test = x_test_pre_transformed
    y_test = y_test_pre_transformed

    with tf.device('/GPU:0'):
        model = my_model(len(np.unique(y)))
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                      metrics=['accuracy'])

        class_weights = {label: 5. if label in registered_labels else 1. for label in np.unique(y)}
        batch_size = 32
        epochs = 250

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=25)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.25)
        model.fit(x, y, batch_size=batch_size, epochs=epochs, class_weight=class_weights,
                  validation_data=(x_test, y_test), validation_batch_size=batch_size,
                  callbacks=[early_stop, reduce_lr])

    model.save('models/latest')
    open('models/latest_label_to_name.json', 'w').write(json.dumps(label_to_name))
    open('models/latest_time.json', 'w').write(json.dumps({'time': time.time()}))
