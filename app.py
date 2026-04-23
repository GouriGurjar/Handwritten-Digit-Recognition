import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# ---------- CONFIG ----------
st.set_page_config(page_title="IntelliDigit AI", layout="wide")

# ---------- PREMIUM CSS ----------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc, #e0f2fe);
    font-family: 'Segoe UI', sans-serif;
}

/* Header */
.header {
    text-align: center;
    padding: 30px 10px;
}

.title {
    font-size: 48px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -1px;
}

.subtitle {
    color: #475569;
    font-size: 17px;
}

/* Divider */
.divider {
    height: 2px;
    background: linear-gradient(to right, transparent, #2563eb, transparent);
    margin: 10px 0 20px 0;
}

/* Glass Card */
.card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    margin-top: 20px;
    transition: 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
}

/* Buttons */
.stButton>button {
    width: 100%;
    border-radius: 14px;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    font-weight: 600;
    padding: 10px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
}

/* Progress bar */
.stProgress > div > div {
    background-color: #2563eb;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="header">
    <div class="title">IntelliDigit</div>
    <div class="subtitle">
        AI-Powered Handwritten Digit Recognition System
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ---------- ABOUT ----------
st.markdown("""
<div class='card'>
<h3>About the System</h3>
<p>
This application uses a Convolutional Neural Network (CNN) trained on handwritten digit data 
to accurately recognize digits from images or real-time drawing input.
</p>
<ul>
<li>Deep Learning Model (TensorFlow/Keras)</li>
<li>Real-time Prediction</li>
<li>Image Processing using OpenCV</li>
<li>Interactive UI using Streamlit</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ---------- WORKFLOW ----------
st.markdown("""
<div class='card'>
<h3>System Workflow</h3>
<ol>
<li>Input image is preprocessed (grayscale, resize, normalization)</li>
<li>Noise removal and digit centering</li>
<li>Image passed to trained CNN model</li>
<li>Prediction generated with probability scores</li>
</ol>
</div>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_my_model():
    return load_model("model.h5")

model = load_my_model()

# ---------- PREPROCESS ----------
def preprocess(img, is_canvas=False):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (28, 28))

    if not is_canvas and np.mean(img) > 127:
        img = 255 - img

    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (20, 20))
        img = cv2.copyMakeBorder(img, 4, 4, 4, 4,
                                 cv2.BORDER_CONSTANT, value=0)

    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img

# ---------- PREDICTION ----------
def show_prediction(pred):
    digit = np.argmax(pred)
    confidence = np.max(pred)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.success(f"Prediction: {digit}")
    st.info(f"Confidence: {confidence*100:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.progress(float(confidence))

    with col2:
        top3 = np.argsort(pred[0])[-3:][::-1]
        st.write("Top Predictions:")
        for i in top3:
            st.write(f"{i} → {pred[0][i]*100:.2f}%")

    st.bar_chart(pred[0])

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- TABS ----------
tab1, tab2 = st.tabs(["Draw Digit", "Upload Image"])

# ---------- DRAW ----------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    canvas = st_canvas(
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
    )

    if st.button("Predict from Canvas"):
        if canvas.image_data is not None:
            with st.spinner("Analyzing..."):
                img = canvas.image_data.astype("uint8")
                img = preprocess(img, is_canvas=True)
                pred = model.predict(img)
                show_prediction(pred)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- UPLOAD ----------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])

    if file is not None:
        image = Image.open(file)
        st.image(image, width=150)

        with st.spinner("Processing..."):
            img = np.array(image)
            img = preprocess(img)
            pred = model.predict(img)
            show_prediction(pred)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- APPLICATIONS ----------
st.markdown("""
<div class='card'>
<h3>Applications</h3>
<ul>
<li>Bank cheque processing</li>
<li>Postal code recognition</li>
<li>Form digitization</li>
<li>Data entry automation</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class='footer'>
<hr>
<p>
AI Application using Deep Learning • Streamlit • OpenCV
</p>
</div>
""", unsafe_allow_html=True)

# ----------Completed----------

