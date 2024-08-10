import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
# Replace 'model.h5' with the path to your model
model = load_model('model.h5')

# Function to preprocess the image


def preprocess_image(img):
    img = img.resize((150, 150))  # Resize to the input size of the model
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array


classes = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# Streamlit app
st.title("Tomato images Disease & Healthy Classification with VGG16")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Make a prediction
    prediction = model.predict(img_array)

    # Display the prediction
    predicted_index = np.argmax(prediction[0])

# Get the predicted class name
    predicted_class = classes[predicted_index]

    # Print the predicted class
    print(f"Predicted class: {predicted_class}")
    st.subheader(f"Predicted class: {predicted_class}")
