import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import requests  # Import requests to handle the download


# --- BACKGROUND IMAGE CSS ---
IMAGE_URL = "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?q=80&w=1965&auto=format&fit=crop"

background_css = f"""
<style>
/* Target the main app container */
.stApp {{
    /* We use a linear-gradient stacked on top of the image 
       to create a dark transparent tint (0.7 opacity). 
       This ensures the white text is readable over the picture.
    */
    background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url("{IMAGE_URL}");
    
    /* Ensure the image covers the whole screen and doesn't repeat */
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center center;
    
    /* Keeps the background fixed while the content scrolls */
    background-attachment: fixed;
}}

/* --- TEXT COLOR ADJUSTMENTS --- */
/* Force nearly all text to be white for contrast against the dark background */
h1, h2, h3, h4, h5, h6, p, div, span, label, li, .stMarkdown {{
    color: #ffffff !important;
}}

/* Adjust file uploader text color specifically */
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {{
     color: #e0e0e0 !important;
}}
</style>
"""

# Inject the CSS into the Streamlit app
st.markdown(background_css, unsafe_allow_html=True)


# --- CONFIGURATION ---
# 🔴 TODO: REPLACE THIS URL with the direct link to your .h5 file from GitHub Releases
MODEL_URL = "https://github.com/Dwaipayan007/DeepFake-Detection-In-Video-Conferencing-Environments-/releases/download/v1.0/deepfake_image_model.h5"

# This is where the file will be saved inside the app's folder
MODEL_LOCAL_PATH = "deepfake_image_model.h5"

# 1. Use caching to prevent the model from reloading (and re-downloading) on every interaction

@st.cache_resource
def load_model():
    # Check if the model file exists locally
    if not os.path.exists(MODEL_LOCAL_PATH):
        # This creates a temporary loading spinner instead of a permanent warning message
        with st.spinner("Downloading model... please wait"):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                
                with open(MODEL_LOCAL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                st.error(f"Failed to download model. Error: {e}")
                st.stop()
            # The spinner automatically disappears here once the block finishes!

    # Load the model
    return tf.keras.models.load_model(MODEL_LOCAL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- MAIN APP UI ---
st.title("Deepfake Detection (Image Only)")
st.write("Upload a face image to check for manipulation.")

file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def preprocess(img):
    # Celeb-DF models often expect RGB. Ensure conversion in case of RGBA/Grayscale
    img = img.convert("RGB") 
    img = img.resize((128, 128))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, 0)

if file:
    # Use columns to make the UI look cleaner
    col1, col2 = st.columns(2)
    
    img = Image.open(file)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(img, use_container_width=True)
    
    with col2:
        st.subheader("Analysis")
        # Run prediction
        try:
            processed_img = preprocess(img)
            prediction_prob = model.predict(processed_img)[0][0]
            
            # Display results with confidence levels
            if prediction_prob < 0.5:
                confidence = (1 - prediction_prob) * 100
                st.success(f"Result: **REAL**")
                st.write(f"Confidence: {confidence:.2f}%")
            else:
                confidence = prediction_prob * 100
                st.error(f"Result: **FAKE**")
                st.write(f"Confidence: {confidence:.2f}%")
                
            # Optional: Progress bar for visual representation
            st.progress(float(prediction_prob))
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    pass










