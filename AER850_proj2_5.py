#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 10:38:11 2023

@author: emilybirkenhead
"""

import os
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

from Aer850_proj2 import create_model



# test_image_folder = '/Users/emilybirkenhead/Documents/4th Yr/AER850 - Intro t oMachine Learning/Project 2/Data/Test/Medium'
# test_image_filename = 'Crack__20180419_06_19_09,915.bmp'  

test_image_folder = '/Users/emilybirkenhead/Documents/4th Yr/AER850 - Intro t oMachine Learning/Project 2/Data/Test/Large'
test_image_filename = 'Crack__20180419_13_29_14,846.bmp'

test_image_path = os.path.join(test_image_folder, test_image_filename)


# Load the pre-trained model
model = create_model(input_shape=(100, 100, 3))  # Adjust input_shape based on your model

# Load the image
img = image.load_img(test_image_path, target_size=(100, 100))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize the image array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension



# Make sure the input shape matches the model's expected input shape
if img_array.shape == (1, 100, 100, 3):  # Adjust the shape based on your model
    # Preprocess the input for the model
    img_array = preprocess_input(img_array)

    # Make a prediction
    prediction = model.predict(img_array)

    class_labels = ['small', 'medium', 'large']
    predicted_class = class_labels[np.argmax(prediction)]

    # Display the image
    plt.imshow(img)
    plt.title(f'Predicted Probabilities: {prediction[0]}')
    plt.show()

    print("Prediction:", prediction)
else:
    print("Input shape does not match the model's expected input shape.")
