from django import views
from django.http import response
from django import forms
from django.views.decorators.csrf import csrf_exempt


class UploadFileForm(forms.Form):
    picture = forms.FileField()


def handle_uploaded_file(f):
    with open('image.jpg', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


@csrf_exempt
def predict(request):
    from pss_recognition.recognition.model_holder import ModelHolder
    from pss_recognition.recognition.recognition import transform_x, get
    import tensorflow as tf
    import numpy as np
    import json

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            model = ModelHolder.get_model()
            label_to_name = json.loads(open('models/latest_label_to_name.json').read())
            handle_uploaded_file(request.FILES['picture'])
            with tf.device('/CPU:0'):
                transformed, _ = transform_x(np.asarray([get('image.jpg')]), np.asarray([0]))
                prediction = model.predict(transformed)
            label = np.argmax(prediction)
            probability = np.max(prediction)
            return response.JsonResponse({'name': label_to_name[str(label)], 'probability': str(probability)})
    return response.HttpResponseBadRequest()
