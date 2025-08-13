import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os

# --- 1. Custom CSS and screen detection for responsive layout ---
def add_custom_css():
    st.markdown("""
        <style>
            /* Main container and background gradient */
            .stApp {
                background-image: linear-gradient(to right, #0a0a0a, #330000);
                color: white;
            }
            /* Headers with a subtle gradient effect */
            h1, h2, h3, h4, h5, h6 {
                color: #ff3333;
                text-shadow: 1px 1px 2px #000000;
            }
            .main-header {
                font-size: 48px !important;
                font-weight: bold;
                text-align: center;
                color: #ff3333;
                text-shadow: 2px 2px 5px #000000;
                animation: fadeIn 2s;
            }
            .subheader {
                font-size: 20px;
                color: #dddddd;
                text-align: center;
                margin-top: -10px;
                animation: fadeIn 3s;
            }
            /* Card-like container for content */
            .st-emotion-cache-1dp5vir {
                background-color: rgba(255, 255, 255, 0.05);
                border: 2px solid #440000;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 5px 5px 15px rgba(0,0,0,0.5);
                transition: all 0.3s ease-in-out;
            }
            .st-emotion-cache-1dp5vir:hover {
                border-color: #ff6666;
                box-shadow: 5px 5px 25px rgba(255,51,51,0.2);
            }
            /* Styling for Streamlit buttons */
            .st-emotion-cache-l99d4s {
                background-color: #ff3333;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                transition: all 0.2s ease-in-out;
            }
            .st-emotion-cache-l99d4s:hover {
                background-color: #aa0000;
                transform: scale(1.05);
            }
            /* Footer */
            .footer {
                text-align: center;
                font-size: 14px;
                color: #888888;
                margin-top: 50px;
            }
            /* Keyframe animation for text fade-in */
            @keyframes fadeIn {
                0% { opacity: 0; }
                100% { opacity: 1; }
            }

            /* Hide Streamlit's default header and footer */
            header[data-testid="stHeader"], footer[data-testid="stFooter"] {
                display: none;
            }
            
            /* --- Mobile-specific styles for responsiveness --- */
            @media screen and (max-width: 768px) {
                .main-header { font-size: 32px !important; }
                .subheader { font-size: 16px; }
            }
        </style>
        """, unsafe_allow_html=True)

# Apply the custom CSS
add_custom_css()

# --- 2. Main App Content with more impressive layout ---
st.markdown("<h1 class='main-header'>🧬 Blood Cell Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Harnessing AI to aid in the detection of different blood cell types.</p>", unsafe_allow_html=True)

# --- 3. Model Information and Loading ---
IMG_SIZE = (224, 224)
CLASS_NAMES = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

@st.cache_resource
def load_model():
    with st.spinner("⏳ Loading AI Model..."):
        time.sleep(2)
        try:
            model = tf.keras.models.load_model('Final-Model05.keras')
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None

model = load_model()

# --- 4. Define Input Widgets and Place them Conditionally ---
sample_images = {
    "TEST IMAGE 01": "sample_images/basophil.jpg",
    "TEST IMAGE 02": "sample_images/lymphocyte.jpg",
    "TEST IMAGE 03": "sample_images/ig.jpg",
    "TEST IMAGE 04": "sample_images/eosinophil.jpg",
    "TEST IMAGE 05": "sample_images/erythroblast.jpg",
    "TEST IMAGE 06": "sample_images/monocyte.jpg",
}

# Use a JavaScript hack to set a session state variable for screen width
# This is a robust way to know if we are on a mobile-like device
js_code = """
<script>
    window.parent.postMessage({
        streamlit: {
            isMobile: window.innerWidth <= 768
        }
    }, "*");
</script>
"""
st.markdown(js_code, unsafe_allow_html=True)

# Check if the session state has been updated
if 'is_mobile' not in st.session_state:
    st.session_state['is_mobile'] = False

# Conditional rendering of input widgets
if st.session_state['is_mobile']:
    # Mobile view: all inputs inside an expanded expander
    with st.expander("🔬 Input Options", expanded=True):
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        st.write("---")
        sample_image_selection = st.selectbox(
            "Or use a sample image", 
            ["--- Select a sample image ---"] + list(sample_images.keys())
        )
else:
    # Desktop view: all inputs inside the sidebar
    with st.sidebar:
        st.header("🔬 Input Options")
        st.markdown("Choose an image to classify.")
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        st.write("---")
        sample_image_selection = st.selectbox(
            "Or use a sample image", 
            ["--- Select a sample image ---"] + list(sample_images.keys())
        )

# --- 5. Image Loading and Prediction Logic ---
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
elif sample_image_selection != "--- Select a sample image ---":
    image_filename = sample_images[sample_image_selection].split('/')[-1]
    sample_image_path = os.path.join(os.path.dirname(__file__), "sample_images", image_filename)
    
    if os.path.exists(sample_image_path):
        image = Image.open(sample_image_path)
        st.image(image, caption=f"Sample: {sample_image_selection}", use_container_width=True)
    else:
        st.warning(f"Sample image '{image_filename}' not found!")
        
def predict(image, model):
    with st.spinner("🔍 Analyzing image..."):
        img = image.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(predictions) * 100
        return predicted_class_name, confidence, predictions[0]

# --- 6. Main Display of Impressive Results ---
if image is not None:
    if model:
        predicted_name, confidence, all_probs = predict(image, model)

        st.markdown("---")
        st.header("Results")
        
        col_pred, col_conf = st.columns(2)
        with col_pred:
            st.metric(label="Predicted Class", value=f"🔬 {predicted_name.upper()}", delta_color="off")
        with col_conf:
            st.metric(label="Confidence", value=f"{confidence:.2f}%", delta_color="off")

        st.markdown("---")
        
        with st.expander("📊 View All Class Probabilities"):
            for i, (name, prob) in enumerate(zip(CLASS_NAMES, all_probs)):
                st.write(f"**{name.capitalize()}**")
                prob_color = "#ff3333" if name == predicted_name else "#888888"
                st.markdown(f"<div style='color:{prob_color}'>{prob * 100:.2f}%</div>", unsafe_allow_html=True)
                st.progress(float(prob))

        st.success(f"**Conclusion:** The model is most confident that the cell is a **{predicted_name}**.")

else:
    st.info("⬆️ To get started, please upload or select a blood cell image.")

# --- 7. Footer ---
st.markdown("<div class='footer'>Created with ❤️ using Streamlit and TensorFlow</div>", unsafe_allow_html=True)
        
