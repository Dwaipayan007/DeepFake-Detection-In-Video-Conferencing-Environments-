import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import requests  # Import requests to handle the download

# --- MATRIX BINARY RAIN ANIMATION (CSS) ---
binary_css = """
<style>
/* 1. The Container - FIXED position to cover the screen */
.binary-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
    /* sitting on top of everything, but invisible to mouse clicks */
    z-index: 99999; 
    pointer-events: none;
}

/* 2. The individual digits */
.binary-container li {
    position: absolute;
    display: block;
    list-style: none;
    
    /* THE MATRIX LOOK */
    color: #0f0; /* Neon Green */
    font-family: 'Courier New', Courier, monospace; /* Monospace font */
    font-weight: bold;
    text-shadow: 0 0 8px rgba(0, 255, 0, 0.8); /* Green glow */
    opacity: 0; /* Start invisible below screen */
    
    /* The movement animation */
    animation: riseUp 15s linear infinite;
    bottom: -50px; /* Start just off-screen at the bottom */
}

/* 3. Keyframes for upward movement */
@keyframes riseUp {
    0% {
        transform: translateY(0);
        opacity: 0;
    }
    10% {
       opacity: 1; /* Fade in quickly at bottom */
    }
    90% {
       opacity: 0.8; /* Stay visible through the middle */
    }
    100% {
        transform: translateY(-110vh); /* Move all the way past the top */
        opacity: 0; /* Fade out at the top */
    }
}

/* 4. Randomizing positions, sizes, and speeds for 20 digits */
/* Slower, bigger digits */
.binary-container li:nth-child(1) { left: 5%; font-size: 30px; animation-duration: 18s; animation-delay: 0s; }
.binary-container li:nth-child(2) { left: 12%; font-size: 24px; animation-duration: 15s; animation-delay: 2s; }
.binary-container li:nth-child(3) { left: 22%; font-size: 28px; animation-duration: 20s; animation-delay: 5s; }
.binary-container li:nth-child(4) { left: 30%; font-size: 22px; animation-duration: 16s; animation-delay: 1s; }
.binary-container li:nth-child(5) { left: 38%; font-size: 32px; animation-duration: 19s; animation-delay: 3s; }
.binary-container li:nth-child(6) { left: 45%; font-size: 26px; animation-duration: 14s; animation-delay: 7s; }
.binary-container li:nth-child(7) { left: 53%; font-size: 30px; animation-duration: 22s; animation-delay: 0s; }
.binary-container li:nth-child(8) { left: 61%; font-size: 24px; animation-duration: 17s; animation-delay: 4s; }
.binary-container li:nth-child(9) { left: 70%; font-size: 28px; animation-duration: 21s; animation-delay: 2s; }
.binary-container li:nth-child(10){ left: 80%; font-size: 22px; animation-duration: 15s; animation-delay: 6s; }

/* Faster, smaller digits to fill gaps */
.binary-container li:nth-child(11){ left: 8%; font-size: 18px; animation-duration: 12s; animation-delay: 9s; }
.binary-container li:nth-child(12){ left: 18%; font-size: 16px; animation-duration: 11s; animation-delay: 1s; }
.binary-container li:nth-child(13){ left: 28%; font-size: 20px; animation-duration: 13s; animation-delay: 4s; }
.binary-container li:nth-child(14){ left: 34%; font-size: 14px; animation-duration: 10s; animation-delay: 8s; }
.binary-container li:nth-child(15){ left: 42%; font-size: 18px; animation-duration: 14s; animation-delay: 2s; }
.binary-container li:nth-child(16){ left: 58%; font-size: 16px; animation-duration: 11s; animation-delay: 5s; }
.binary-container li:nth-child(17){ left: 66%; font-size: 20px; animation-duration: 13s; animation-delay: 1s; }
.binary-container li:nth-child(18){ left: 75%; font-size: 14px; animation-duration: 10s; animation-delay: 3s; }
.binary-container li:nth-child(19){ left: 88%; font-size: 18px; animation-duration: 12s; animation-delay: 7s; }
.binary-container li:nth-child(20){ left: 95%; font-size: 16px; animation-duration: 11s; animation-delay: 0s; }

</style>

<ul class="binary-container">
    <li>0</li><li>1</li><li>0</li><li>0</li><li>1</li>
    <li>1</li><li>0</li><li>1</li><li>0</li><li>1</li>
    <li>0</li><li>1</li><li>1</li><li>0</li><li>0</li>
    <li>1</li><li>0</li><li>0</li><li>1</li><li>1</li>
</ul>
"""

# Inject the CSS/HTML into Streamlit
st.markdown(binary_css, unsafe_allow_html=True)

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









