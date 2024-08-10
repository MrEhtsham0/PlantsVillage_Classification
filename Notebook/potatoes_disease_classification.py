# -*- coding: utf-8 -*-
"""Potatoes_Disease_Classification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZqvRJyICO7eOhCAoeSyVBDVKb22cAnjV
"""

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
#-p flag ensures that the command creates parent directories as needed,
#and it won't throw an error if the directory already exists.

!kaggle datasets download -d emmarex/plantdisease

import zipfile
zip_ref = zipfile.ZipFile('/content/plantdisease.zip')
zip_ref.extractall('/content')
zip_ref.close()

import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16

input_shape = (150, 150, 3)
input_tensor = keras.Input(shape=input_shape)
conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_tensor=input_tensor
    )

conv_base.trainable=True
set_trainable=False
for layer in conv_base.layers:
  if layer.name=='block5_conv1':
    set_trainable=True
  if set_trainable:
    layer.trainable=True
  else:
    layer.trainable=False

for layers in conv_base.layers:
  print(layers.name,layers.trainable)

conv_base.summary()

x = conv_base.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(15, activation='softmax')(x)

model = keras.Model(inputs=input_tensor, outputs=x)

print(model.summary())

import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
original_data_dir = '/content/PlantVillage'
base_dir = '/content/'

# Create directories for train, validation, and test sets
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Set the split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# List of class folders
classes = [
 'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# Splitting the data
for cls in classes:
    class_dir = os.path.join(original_data_dir, cls)
    images = os.listdir(class_dir)

    # Split the images into train, val, and test
    train_images, temp_images = train_test_split(images, train_size=train_ratio, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=test_ratio/(test_ratio + val_ratio), random_state=42)

    # Move images to train folder
    train_class_dir = os.path.join(train_dir, cls)
    os.makedirs(train_class_dir, exist_ok=True)
    for img in train_images:
        shutil.move(os.path.join(class_dir, img), os.path.join(train_class_dir, img))

    # Move images to validation folder
    val_class_dir = os.path.join(val_dir, cls)
    os.makedirs(val_class_dir, exist_ok=True)
    for img in val_images:
        shutil.move(os.path.join(class_dir, img), os.path.join(val_class_dir, img))

    # Move images to test folder
    test_class_dir = os.path.join(test_dir, cls)
    os.makedirs(test_class_dir, exist_ok=True)
    for img in test_images:
        shutil.move(os.path.join(class_dir, img), os.path.join(test_class_dir, img))

print("Data splitting is complete.")

train_ds=keras.utils.image_dataset_from_directory(
    directory='train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(150,150)
)
test_ds=keras.utils.image_dataset_from_directory(
    directory='test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(150,150)
)
val_ds=keras.utils.image_dataset_from_directory(
    directory='val',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(150,150)
)

def process(image, label):
  image = tensorflow.cast(image/255.0, tensorflow.float32)
  return image, label

train_ds = train_ds.map(process)
test_ds = test_ds.map(process)
val_ds = val_ds.map(process)

# If your labels are integers (0 to 14), use this:
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=5, validation_data=val_ds)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')

import numpy as np
from tensorflow.keras.preprocessing import image

# Load and preprocess the image
img_path = '/content/0031da2a-8edd-468f-a8b1-106657717a32___RS_HL 0105.JPG'  # Replace with your image path
img = image.load_img(img_path, target_size=(150, 150))  # Resize to 150x150
img_array = image.img_to_array(img)  # Convert to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize the image (same as during training)

# Make prediction
prediction = model.predict(img_array)

# List of class names
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
               'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot',
               'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
               'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
               'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
               'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

# Find the index of the class with the highest probability
predicted_index = np.argmax(prediction[0])

# Get the predicted class name
predicted_class = class_names[predicted_index]

# Print the predicted class
print(f"Predicted class: {predicted_class}")

# Save the entire model to a single file
model.save('model.h5')

from tensorflow.keras.models import load_model
loaded_model = load_model('model.h5')

