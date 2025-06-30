from django.shortcuts import render

# Create your views here.
import os
import numpy as np
from django.shortcuts import render
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your model
model = load_model(os.path.join(settings.BASE_DIR, 'classifier/flower_model.h5'))
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']  # Change to your actual class names

def predict_image(request):
    prediction = None
    image_url = None

    if request.method == 'POST' and 'image' in request.FILES:
        img_file = request.FILES['image']
        img_path = os.path.join(settings.MEDIA_ROOT, img_file.name)

        with open(img_path, 'wb+') as f:
            for chunk in img_file.chunks():
                f.write(chunk)

        img = image.load_img(img_path, target_size=(180, 180))  # use your model's input size
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        result = model.predict(img_array)
        prediction = classes[np.argmax(result)]
        image_url = settings.MEDIA_URL + img_file.name

    return render(request, 'classifier/index.html', {
        'prediction': prediction,
        'image_url': image_url
    })
