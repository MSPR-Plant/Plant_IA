import io
from urllib import request

import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('ressources/vgg19.h5')

image_bytes = request.files['image'].read()
image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
image = image.resize((224, 224))
image_array = np.array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

predictions = model.predict(image_array)
predicted_class = np.argmax(predictions[0])

print(predicted_class)
