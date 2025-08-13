import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os

# --- 1. Custom CSS and styling for the app ---
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

            /* --- Mobile-specific styles --- */
            @media screen and (max-width: 768px) {
                .main-header { font-size: 32px !important; }
                .subheader { font-size: 16px; }
                /* Hide the sidebar on mobile */
                [data-testid="stSidebar"] { display: none; }
                /* Change layout for mobile */
                [data-testid="stColumn"] { width: 100% !important; }
                /* Show the input expander on mobile */
                [data-testid="stExpander"] { display: block !important; }
            }
            
            /* --- Desktop-specific styles --- */
            @media screen and (min-width: 769px) {
                /* Hide the input expander on desktop */
                [data-testid="stExpander"] { display: none; }
            }

            /* --- Custom styling for the new result metrics --- */
            .metric-card {
                background-color: rgba(255, 255, 255, 0.05);
                border: 2px solid #ff3333;
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                margin-bottom: 20px;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.5);
            }
            .metric-label {
                font-size: 16px;
                color: #ffcccc;
                font-weight: bold;
            }
            .metric-value {
                font-size: 24px;
                color: #ffffff;
                font-weight: bold;
            }
        </style>
        """, unsafe_allow_html=True)

# Apply the custom CSS
add_custom_css()

# --- 2. Main App Content ---
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

# --- 4. Define Input Widgets and Image Previews ---
sample_images = {
    "Basophil": "sample_images/basophil.jpg",
    "Lymphocyte": "sample_images/lymphocyte.jpg",
    "IG": "sample_images/ig.jpg",
    "Eosinophil": "sample_images/eosinophil.jpg",
    "Erythroblast": "sample_images/erythroblast.jpg",
    "Monocyte": "sample_images/monocyte.jpg",
}

uploaded_file = None
sample_image_selection = None
image_to_show = None

# Sidebar for desktop
with st.sidebar:
    st.header("🔬 Input Options")
    st.markdown("Choose an image to classify.")
    
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    st.write("---")
    sample_image_selection = st.selectbox(
        "Or use a sample image", 
        ["--- Select a sample image ---"] + list(sample_images.keys())
    )
    
    if uploaded_file is not None:
        image_to_show = Image.open(uploaded_file)
        st.image(image_to_show, caption="Uploaded Image", use_container_width=True)
    elif sample_image_selection != "--- Select a sample image ---":
        image_filename = sample_images[sample_image_selection].split('/')[-1]
        sample_image_path = os.path.join(os.path.dirname(__file__), "sample_images", image_filename)
        if os.path.exists(sample_image_path):
            image_to_show = Image.open(sample_image_path)
            st.image(image_to_show, caption=f"Sample: {sample_image_selection}", use_container_width=True)

# Expander for mobile
with st.expander("🔬 Input Options", expanded=True):
    uploaded_file_expander = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="expander_uploader")
    st.write("---")
    sample_image_selection_expander = st.selectbox(
        "Or use a sample image", 
        ["--- Select a sample image ---"] + list(sample_images.keys()),
        key="expander_selectbox"
    )
    
    if uploaded_file_expander is not None:
        image_to_show = Image.open(uploaded_file_expander)
        st.image(image_to_show, caption="Uploaded Image", width=250)
    elif sample_image_selection_expander != "--- Select a sample image ---":
        image_filename = sample_images[sample_image_selection_expander].split('/')[-1]
        sample_image_path = os.path.join(os.path.dirname(__file__), "sample_images", image_filename)
        if os.path.exists(sample_image_path):
            image_to_show = Image.open(sample_image_path)
            st.image(image_to_show, caption=f"Sample: {sample_image_selection_expander}", width=250)

# The uploaded_file and sample_image_selection variables need to be correctly
# captured from whichever input source is active.
uploaded_file = uploaded_file or uploaded_file_expander
sample_image_selection = sample_image_selection or sample_image_selection_expander

# --- 5. Prediction Logic ---
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
if image_to_show is not None:
    if model:
        predicted_name, confidence, all_probs = predict(image_to_show, model)

        st.markdown("---")
        st.header("Results")
        
        col_pred, col_conf = st.columns(2)
        with col_pred:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Class</div>
                    <div class="metric-value">🔬 {predicted_name.upper()}</div>
                </div>
            """, unsafe_allow_html=True)
        with col_conf:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{confidence:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)

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
