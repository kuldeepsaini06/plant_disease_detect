# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 12:45:39 2023

@author: ASUS
"""
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf
from PIL import Image, ImageOps

#@st.cache(allow_output_mutation=True)

# Load model
model= load_model("plant_disease_class.h5")

# Names Of Classes
Class_Names=["Apple___Cedar_apple_rust","Corn_(maize)___Common_rust_","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"]

# Setting title
st.title("Plant Disease Detection")

st.markdown("Upload an image of the plant leaf")

#Uploading the plant image

plant_image= st.file_uploader("Choose an image", type=["Jpg","png"])

submit= st.button("Predict")

if submit:
    
    if plant_image is not None:
    # Read the image file
        image = Image.open(plant_image)
        st.image(image)

    # Resize the image to the required input shape of the model
        image = image.resize((32, 32))
    
    # Convert the image to grayscale
        image = image.convert("L")

    # Convert the image to a numpy array
        image_array = np.array(image)

    # Normalize the image array
        image_array = image_array / 255.0

    # Add an extra dimension to match the model's input shape
        image_array = np.transpose(image_array, (1, 0))
        image_array = np.expand_dims(image_array, axis=0)

    # Perform prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)

    # Display the predicted class label
        st.write(Class_Names[predicted_class])
        st.balloons()
        
else:
    pass


        
        
        
        







