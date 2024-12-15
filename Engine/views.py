from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import base64
from tensorflow.keras.models import load_model


# Load the model and ASCII map
model = load_model('Assets\Model\handwrittenmodel.h5', compile=True)
ascii_map = pd.read_csv("Assets\Model\mapping.csv")

# Function to render the main HTML page
def index(request):
    return render(request, "Engine/index.html")

@csrf_exempt  # Exempt CSRF for this view (required for POST requests without CSRF token)
def predict(request):
    if request.method == "POST":
        try:
            canvasdata = request.POST.get('canvasimg')
            encoded_data = canvasdata.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_LINEAR)
            gray_image = gray_image / 255.0

            # Expand dimensions to fit model input shape
            gray_image = np.expand_dims(gray_image, axis=-1)
            img = np.expand_dims(gray_image, axis=0)

            # Predict using the model
            prediction = model.predict(img)
            cl = list(prediction[0])
            predicted_character = ascii_map["Character"][cl.index(max(cl))]

            return render(request, "Engine/index.html", {"value": predicted_character})
        except Exception as e:
            return JsonResponse({"error": str(e)})

    return JsonResponse({"error": "Invalid request method"})



