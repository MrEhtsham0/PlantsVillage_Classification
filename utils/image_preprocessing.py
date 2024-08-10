from keras.preprocessing import image
import numpy as np


def preprocess_image(img):
    img = img.resize((150, 150))  # Resize to the input size of the model
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array
