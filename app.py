import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import requests  # Import requests to handle the download

# --- MASTER CSS: BACKGROUND + ANIMATION ---
master_css = """
<style>
/* 1. MAIN BACKGROUND IMAGE */
.stApp {
    background-image: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url("https://images.unsplash.com/photo-1620712943543-bcc4688e7485?q=80&w=1965&auto=format&fit=crop");
    background-size: cover;
    background-attachment: fixed;
}

/* 2. TEXT COLOR FIXES */
h1, h2, h3, p, span, div {
    color: white !important;
}

/* 3. MATRIX ANIMATION CONTAINER */
.matrix-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none; /* Allows clicks to pass through */
    z-index: 999999; /* Forces it to the very front */
     
}

/* 4. MATRIX DIGITS */
.matrix-container li {
    position: absolute;
    display: block;
    list-style: none;
    color: #0f0; /* Neon Green */
    font-size: 20px;
    font-weight: bold;
    font-family: monospace;
    text-shadow: 0 0 5px #0f0;
    
    /* Animation settings */
    animation: riseUp 10s linear infinite;
    bottom: -50px; /* Start below screen */
}

/* 5. MOVEMENT KEYFRAMES (UPWARDS) */
@keyframes riseUp {
    0% {
        transform: translateY(0);
        opacity: 0;
    }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% {
        transform: translateY(-110vh); /* Move past top of screen */
        opacity: 0;
    }
}

/* 6. RANDOMIZED DIGITS */
.matrix-container li:nth-child(1) { left: 10%; animation-duration: 15s; animation-delay: 0s; }
.matrix-container li:nth-child(2) { left: 20%; animation-duration: 12s; animation-delay: 2s; }
.matrix-container li:nth-child(3) { left: 30%; animation-duration: 18s; animation-delay: 4s; }
.matrix-container li:nth-child(4) { left: 40%; animation-duration: 14s; animation-delay: 1s; }
.matrix-container li:nth-child(5) { left: 50%; animation-duration: 16s; animation-delay: 3s; }
.matrix-container li:nth-child(6) { left: 60%; animation-duration: 13s; animation-delay: 5s; }
.matrix-container li:nth-child(7) { left: 70%; animation-duration: 19s; animation-delay: 2s; }
.matrix-container li:nth-child(8) { left: 80%; animation-duration: 15s; animation-delay: 6s; }
.matrix-container li:nth-child(9) { left: 90%; animation-duration: 17s; animation-delay: 1s; }
.matrix-container li:nth-child(10){ left: 5%;  animation-duration: 14s; animation-delay: 4s; }
</style>

<ul class="matrix-container">
    <li>1</li><li>0</li><li>1</li><li>0</li><li>1</li>
    <li>0</li><li>1</li><li>0</li><li>1</li><li>0</li>
</ul>
"""

# INJECT INTO STREAMLIT
st.markdown(master_css, unsafe_allow_html=True)

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











