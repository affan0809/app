import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cats_dogs_model.h5")
    return model

model = load_model()

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((160, 160))  # Match input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

# Streamlit app interface
st.title("Cat vs Dog Classifier")
st.write("Upload an image to predict whether it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
     
    if st.button("Predict"):
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        pred_class = "Dog" if prediction[0][0] > 0.5 else "Cat"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        st.markdown(f"### Prediction: {pred_class} ({confidence:.2%} confidence)")