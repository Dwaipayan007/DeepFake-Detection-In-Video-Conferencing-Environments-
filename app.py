import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import requests  # Import requests to handle the download

# --- CONFIGURATION ---
# 🔴 TODO: REPLACE THIS URL with the direct link to your .h5 file from GitHub Releases
# It usually looks like: https://github.com/username/repo/releases/download/v1.0/deepfake_image_model.h5
MODEL_URL = "import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import requests  # Import requests to handle the download

# --- CONFIGURATION ---
# 🔴 TODO: REPLACE THIS URL with the direct link to your .h5 file from GitHub Releases
# It usually looks like: https://github.com/username/repo/releases/download/v1.0/deepfake_image_model.h5
MODEL_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/releases/download/v1.0/deepfake_image_model.h5"

# This is where the file will be saved inside the app's folder
MODEL_LOCAL_PATH = "deepfake_image_model.h5"

# 1. Use caching to prevent the model from reloading (and re-downloading) on every interaction
@st.cache_resource
def load_model():
    # Check if the model file exists locally
    if not os.path.exists(MODEL_LOCAL_PATH):
        st.warning("Model file not found locally. Downloading from GitHub Releases... (This may take a minute)")
        try:
            # Send a request to get the file
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()  # Check for download errors
            
            # Write the file to the local system
            with open(MODEL_LOCAL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Download complete! Loading model...")
        except Exception as e:
            st.error(f"Failed to download model. Error: {e}")
            st.error("Please check your MODEL_URL in the code and ensure the file exists in GitHub Releases.")
            st.stop()

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
    pass"

# This is where the file will be saved inside the app's folder
MODEL_LOCAL_PATH = "deepfake_image_model.h5"

# 1. Use caching to prevent the model from reloading (and re-downloading) on every interaction
@st.cache_resource
def load_model():
    # Check if the model file exists locally
    if not os.path.exists(MODEL_LOCAL_PATH):
        st.warning("Model file not found locally. Downloading from GitHub Releases... (This may take a minute)")
        try:
            # Send a request to get the file
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()  # Check for download errors
            
            # Write the file to the local system
            with open(MODEL_LOCAL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Download complete! Loading model...")
        except Exception as e:
            st.error(f"Failed to download model. Error: {e}")
            st.error("Please check your MODEL_URL in the code and ensure the file exists in GitHub Releases.")
            st.stop()

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
