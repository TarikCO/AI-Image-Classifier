import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from PIL import Image


# Logic of the Code --------------

def load_model():
    model = MobileNetV2(weights="imagenet") # Create model in form of percentages
    return model

def preprocess_image(image):
    image = image.convert("RGB") # ensure 3 channels
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def classify_image(model, image): # Process the image and decode the percentages into classifications
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f'Error classifying the image: {str(e)}')
        return None
        

# User interface --------------

def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered")

    st.title("AI Image Classifier")
    st.write("Upload an image and the AI will predict what is in it.")

    # cache the function to not load it over and over again
    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = st.image(
            uploaded_file, caption="Uploaded Image", use_container_width=True
        )
        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Analyzing Image..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Predictions")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")

if __name__ == "__main__":
    main()