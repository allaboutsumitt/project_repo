# app.py
import os, json, numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="centered")

# Choose which model to serve
# MODEL_PATH = "models/fruit_cnn.keras"
MODEL_PATH = "models/fruit_resnet50_ft.keras"  # switch to cnn if that's what you trained
CLASS_MAP_PATH = "class_indices.json"

@st.cache_resource
def load_keras_model(path):
    return load_model(path)

@st.cache_data
def load_class_names(json_path):
    with open(json_path, "r") as f:
        idx_to_class = json.load(f)  # {"0": "Apple", ...}
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    return [idx_to_class[i] for i in sorted(idx_to_class)]

model = load_keras_model(MODEL_PATH)
class_names = load_class_names(CLASS_MAP_PATH)

# Infer input size from model
input_h, input_w = model.input_shape[1], model.input_shape[2]
IMG_SIZE = (input_w, input_h)

st.title("🍓 Fruit Image Classifier")
st.caption("Upload a fruit image or use your camera. Trained on a 10-class Fruits-360 subset.")

def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img) / 255.0
    return np.expand_dims(x, axis=0)

def predict(pil_img):
    x = preprocess(pil_img)
    probs = model.predict(x)[0]
    top_idx = np.argsort(probs)[::-1]
    return [(class_names[i], float(probs[i])) for i in top_idx[:3]]

tab1, tab2 = st.tabs(["Upload Image", "Use Camera"])

with tab1:
    uploaded = st.file_uploader("Upload JPG/PNG", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_column_width=True)
        top3 = predict(img)
        st.subheader(f"Prediction: {top3[0][0]} ({top3[0][1]*100:.1f}%)")
        with st.expander("Top-3 probabilities"):
            for name, p in top3:
                st.write(f"- {name}: {p*100:.1f}%")

with tab2:
    cam = st.camera_input("Take a photo")
    if cam:
        img = Image.open(cam)
        st.image(img, caption="Captured image", use_column_width=True)
        top3 = predict(img)
        st.subheader(f"Prediction: {top3[0][0]} ({top3[0][1]*100:.1f}%)")
        with st.expander("Top-3 probabilities"):
            for name, p in top3:
                st.write(f"- {name}: {p*100:.1f}%")

st.markdown("---")
st.caption("Tip: If predictions look off, ensure your model and class_indices.json came from the same training run.")