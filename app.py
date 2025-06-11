# app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load models
@st.cache_resource
def load_model_densenet():
    return tf.keras.models.load_model('model/custard_model_densenet121.h5')

@st.cache_resource
def load_model_xception():
    return tf.keras.models.load_model('model/custard_model_xception.h5')

# Class names → ตาม model output
class_names_raw = [
    'Anthracnose',
    'Black canker',
    'Cylindrocladium leaf spot - fruit',
    'Cylindrocladium leaf spot - leaf',
    'Diplodia rot',
    'Mealy bug'
]

# Mapping → ชื่อที่จะโชว์ใน App
class_display_name = {
    'Anthracnose': 'Anthracnose',
    'Black canker': 'Black canker',
    'Cylindrocladium leaf spot - fruit': 'Cylindrocladium leaf spot',
    'Cylindrocladium leaf spot - leaf': 'Cylindrocladium leaf spot',
    'Diplodia rot': 'Diplodia rot',
    'Mealy bug': 'Mealy bug'
}

# Title
st.title("Custard Apple Disease Detection Web App")
st.write("Upload an image of custard apple fruit or leaf and select the model for disease classification.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Model selection
model_choice = st.radio(
    "Select Model:",
    ("DenseNet121", "Xception")
)

# Predict button
if uploaded_file is not None:
    image_data = Image.open(uploaded_file)
    
    # Show image → no caption → small size
    st.image(image_data, use_container_width=False, width=300)

    # Prepare image for model
    img = image_data.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Load selected model
    if model_choice == "DenseNet121":
        model = load_model_densenet()
    else:
        model = load_model_xception()

    # Predict
    prediction = model.predict(img_array)[0]  # เอา array 1D ออกมา
    top_indices = prediction.argsort()[-6:][::-1]  # เอาทุก class มาเรียงก่อน (ป้องกัน mapping ซ้ำ)

    # เตรียม Dict สำหรับเก็บ class display name + confidence
    display_confidence = {}

    for idx in top_indices:
        class_name_raw = class_names_raw[idx]
        class_name_display = class_display_name[class_name_raw]
        confidence = prediction[idx] * 100

        # ถ้ายังไม่มี class นี้ → ใส่ลงไป
        if class_name_display not in display_confidence:
            display_confidence[class_name_display] = confidence
        else:
            # ถ้ามีแล้ว (เช่น leaf/fruit mapping เป็นชื่อเดียวกัน) → เอาค่า confidence สูงสุด
            display_confidence[class_name_display] = max(display_confidence[class_name_display], confidence)

    # เอามาเรียง Top-3
    display_confidence_sorted = sorted(display_confidence.items(), key=lambda x: x[1], reverse=True)[:3]

    # Show Top-3 Predictions
    st.info("**Top-3 Predicted Classes:**")
    for class_name_display, confidence in display_confidence_sorted:
        st.write(f"- {class_name_display}: **{confidence:.2f}%**")
