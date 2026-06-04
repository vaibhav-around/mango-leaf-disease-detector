import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Cache the model so it only loads once per session
@st.cache_resource
def load_verifier_model():
    return tf.keras.models.load_model("mango_leaf_verification.keras")


@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model("mango_leaf_disease_model.keras")


verifier_model = load_verifier_model()
disease_model = load_disease_model()


def verify_mango_leaf(img_array):
    pred = verifier_model.predict(img_array, verbose=0)[0][0]
    return pred


class_names = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould",
]

st.set_page_config(page_title="Mango Leaf Disease Detection", page_icon="🍃")

st.title("🍃 Mango Leaf Disease Detection")

st.write("Upload a mango leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize to match training size
    img = image.resize((224, 224))
    img_array = np.array(img)

    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    # Handle RGBA images
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Expand dims for model input (batch size of 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    verification_score = verify_mango_leaf(img_array)
    st.write(f"Verifier Score: {verification_score:.4f}")

    if verification_score > 0.85:
        st.error("❌ This image does not appear to be a mango leaf.")
    else:
        prediction = disease_model.predict(img_array, verbose=0)

        probs = prediction[0]

        predicted_index = np.argmax(probs)

        predicted_class = class_names[predicted_index]

        confidence = float(np.max(probs) * 100)

        if confidence < 60:
            st.warning(
                "⚠️ Model is uncertain. Please upload a clearer mango leaf image."
            )
        else: 
             st.success(
                f"Prediction: {predicted_class}"
                )

             st.write(
                f"Confidence: {confidence:.2f}%"
                )

             st.subheader(
                "Top 3 Predictions"
              )

             top_indices = np.argsort(
                probs
              )[::-1][:3]

             for idx in top_indices:

                st.write(
                    f"{class_names[idx]} : {probs[idx]*100:.2f}%"
                )
