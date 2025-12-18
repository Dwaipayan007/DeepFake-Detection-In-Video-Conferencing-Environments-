import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import requests  # Import requests to handle the download
import random

# --- MASTER CSS: FULL SCREEN MATRIX RAIN ---
# We generate the CSS programmatically to save space and make it truly random
particles = ""
css_rules = ""

# Create 40 random particles
for i in range(1, 41):
    # Random position across the screen (0% to 100%)
    left_pos = random.randint(1, 99) 
    # Random speed (10s to 25s)
    duration = random.randint(10, 25)
    # Random delay so they don't start all at once (0s to 15s)
    delay = random.randint(0, 15)
    # Random size for depth effect
    size = random.randint(15, 25)
    
    # 0 or 1
    digit = random.choice(["0", "1"])
    
    # Build the HTML list item
    particles += f"<li>{digit}</li>"
    
    # Build the CSS rule for this specific particle
    css_rules += f"""
    .matrix-container li:nth-child({i}) {{
        left: {left_pos}%;
        font-size: {size}px;
        animation-duration: {duration}s;
        animation-delay: {delay}s;
    }}
    """

master_css = f"""
<style>
/* 1. MAIN BACKGROUND IMAGE */
.stApp {{
    background-image: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url("https://images.unsplash.com/photo-1620712943543-bcc4688e7485?q=80&w=1965&auto=format&fit=crop");
    background-size: cover;
    background-attachment: fixed;
}}

/* 2. TEXT COLORS */
h1, h2, h3, p, span, div, label {{
    color: white !important;
}}

/* 3. ANIMATION CONTAINER */
.matrix-container {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 999999;
    overflow: hidden;
}}

/* 4. BASE PARTICLE STYLE */
.matrix-container li {{
    position: absolute;
    display: block;
    list-style: none;
    color: #0f0; /* Neon Green */
    font-weight: bold;
    font-family: monospace;
    text-shadow: 0 0 5px #0f0;
    opacity: 0;
    bottom: -50px; /* Start below screen */
    animation-name: riseUp;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
}}

/* 5. MOVEMENT KEYFRAMES */
@keyframes riseUp {{
    0% {{
        transform: translateY(0);
        opacity: 0;
    }}
    10% {{ opacity: 0.8; }}
    90% {{ opacity: 0.8; }}
    100% {{
        transform: translateY(-110vh);
        opacity: 0;
    }}
}}

/* 6. INJECT GENERATED RANDOM RULES */
{css_rules}

</style>

<ul class="matrix-container">
    {particles}
</ul>
"""

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












