import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model("mango_leaf_disease_model.keras")

# IMPORTANT:
# Replace this list with the EXACT output of:
# print(class_names)
# from your training notebook

class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

st.set_page_config(
    page_title="Mango Leaf Disease Detection",
    page_icon="🍃"
)

st.title("🍃 Mango Leaf Disease Detection")

st.write(
    "Upload a mango leaf image and the model will predict the disease."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize to match training size
    img = image.resize((224, 224))

    img_array = np.array(img)

    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    # Handle RGBA images
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction[0])

    predicted_class = class_names[predicted_index]

    confidence = float(np.max(prediction[0]) * 100)

    st.success(
        f"Prediction: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence:.2f}%"
    )