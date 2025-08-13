import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os

# --- 1. Custom CSS for an impressive, dynamic look ---
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
            
             /* Hide the main header and footer from Streamlit's default theme */
            header[data-testid="stHeader"] {
                display: none;
            }
            .st-emotion-cache-1dp5vir { /* Or the specific selector for the header container */
                display: none;
            }
            
            /* ... your existing CSS rules ... */
        </style>
        """, unsafe_allow_html=True)

# Apply the custom CSS
add_custom_css()

# --- 2. Main App Content with more impressive layout ---
st.markdown("<h1 class='main-header'>🧬 Blood Cell Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Harnessing AI to aid in the detection of different blood cell types.</p>", unsafe_allow_html=True)

# --- 3. Model Information and Loading ---
#MODEL_PATH = r"C:\Users\admin\Btech Sem 6\BloodCancerDetection\Blood Cancer Detection\Detection Model\BCD-Streamlit\Final-Model05.keras"
# Correct way to load the model from the same directory
#import tensorflow as tf
model = tf.keras.models.load_model('Final-Model05.keras')

IMG_SIZE = (224, 224)
CLASS_NAMES = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

@st.cache_resource
def load_model():
    with st.spinner("⏳ Loading AI Model..."):
        time.sleep(2) # Simulate network delay
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None

model = load_model()

# --- 4. Sidebar for Interactive Elements ---
st.sidebar.header("🔬 Input Options")
st.sidebar.markdown("Choose an image to classify.")

# Updated dictionary with correct file extensions
sample_images = {
    "Basophil": "sample_images/basophil.jpg",
    "Lymphocyte": "sample_images/lymphocyte.jpg",
    "IG": "sample_images/ig.jpg",
    "Eosinophil": "sample_images/eosinophil.jpg",
    "Erythroblast": "sample_images/erythroblast.jpg",
    "Monocyte": "sample_images/monocyte.jpg",
}

uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
st.sidebar.write("---")
sample_image_selection = st.sidebar.selectbox(
    "Or use a sample image", 
    ["--- Select a sample image ---"] + list(sample_images.keys())
)

# --- 5. Image Loading and Prediction Logic ---
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Uploaded Image", use_container_width=True)
elif sample_image_selection != "--- Select a sample image ---":
    # --- CORRECTED LINE BELOW ---
    # We now look for the correct .jpg extension based on your dictionary
    image_filename = sample_images[sample_image_selection].split('/')[-1]
    sample_image_path = os.path.join(os.path.dirname(__file__), "sample_images", image_filename)
    
    if os.path.exists(sample_image_path):
        image = Image.open(sample_image_path)
        st.sidebar.image(image, caption=f"Sample: {sample_image_selection}", use_container_width=True)
    else:
        st.sidebar.warning(f"Sample image '{image_filename}' not found!")
        
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
        
        # Display the main prediction using a metric
        col_pred, col_conf = st.columns(2)
        with col_pred:
            st.metric(label="Predicted Class", value=f"🔬 {predicted_name.upper()}", delta_color="off")
        with col_conf:
            st.metric(label="Confidence", value=f"{confidence:.2f}%", delta_color="off")

        st.markdown("---")
        
        # Use an expander for detailed probabilities
        with st.expander("📊 View All Class Probabilities"):
            for i, (name, prob) in enumerate(zip(CLASS_NAMES, all_probs)):
                st.write(f"**{name.capitalize()}**")
                # Highlight the predicted class with a different color
                prob_color = "#ff3333" if name == predicted_name else "#888888"
                st.markdown(f"<div style='color:{prob_color}'>{prob * 100:.2f}%</div>", unsafe_allow_html=True)
                st.progress(float(prob))

        # A final conclusion
        st.success(f"**Conclusion:** The model is most confident that the cell is a **{predicted_name}**.")

else:
    st.info("⬆️ To get started, please upload or select a blood cell image.")

# --- 7. Footer ---
st.markdown("<div class='footer'>Created with ❤️ using Streamlit and TensorFlow</div>", unsafe_allow_html=True)
