import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# 1. Use caching to prevent the model from reloading on every interaction
@st.cache_resource
def load_model():
    # Ensure the path matches your folder structure
    return tf.keras.models.load_model("model/deepfake_image_model.h5")

model = load_model()

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
if __name__ == "__main__":
    pass
