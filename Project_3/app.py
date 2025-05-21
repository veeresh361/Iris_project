import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from pca_utils import (
    load_images_from_folder,
    apply_pca,
    save_pca_model,
    load_pca_model,
    reconstruct_image
)

from plotting_utils import (
    plot_explained_variance,
    show_eigenfaces,
    show_image_comparison,
    show_image,
    show_reconstructed_image
)

from pca_config import MODEL_SAVE_PATH, IMAGE_FOLDER

# Page configuration
st.set_page_config(page_title="PCA Image Explorer", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    h1 {
        color: #4B8BBE;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #4B8BBE 0%, #306998 100%);
        border: none;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
for key, default in {
    'X': None,
    'Show Eigen Faces': False,
    'eigen_faces': None,
    'reconstruct_button': False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Page Title
st.title("🧠 PCA Image Explorer")

# Sidebar Settings
st.sidebar.title("🔧 Settings")
available_sizes = [64, 128, 256, 512]
st.session_state['img_width'] = st.sidebar.selectbox("Image Width", available_sizes, index=1)
st.session_state['img_height'] = st.sidebar.selectbox("Image Height", available_sizes, index=1)

# Load Images
if st.button('Load Images'):
    with st.spinner("Loading images..."):
        st.session_state['X'], image_shape = load_images_from_folder(
            IMAGE_FOLDER,
            image_size=(st.session_state['img_width'], st.session_state['img_height'])
        )
        if st.session_state['X'].size == 0:
            st.error("❌ No images found in the folder!")
            st.stop()
        st.success("✅ Images are Loaded")

# Show Sample Images
no_of_images = st.selectbox("Select No of Images", [0, 5, 10])
if no_of_images != 0 and st.session_state['X'] is not None:
    st.subheader("📷 Sample Image from Dataset")
    figure = show_image(st.session_state['X'], no_of_images,
                        (st.session_state['img_height'], st.session_state['img_width']))
    st.pyplot(figure)

# Run PCA
if st.sidebar.button("Run PCA"):
    st.session_state['pca'], X_pca = apply_pca(st.session_state['X'])
    save_pca_model(st.session_state['pca'], MODEL_SAVE_PATH)
    st.success("✅ PCA Model Trained and Saved!")

    st.subheader("Explained Variance Ratio")
    variance_ratio_fig, st.session_state['optimal_components'] = plot_explained_variance(
        st.session_state['pca']
    )
    st.pyplot(variance_ratio_fig)

# Show Eigen Faces
if st.sidebar.button("Show Eigen Faces"):
    st.session_state['Show Eigen Faces'] = True

if st.session_state['Show Eigen Faces']:
    st.session_state['eigen_faces'] = st.selectbox(
        "Select No of Images", [0, 5, 10], key="Eigen Face Selector"
    )
    if st.session_state['eigen_faces'] != 0:
        image_shape = (st.session_state['img_height'], st.session_state['img_width'])
        eigen_fig = show_eigenfaces(
            st.session_state['pca'], image_shape, n_components=st.session_state['eigen_faces']
        )
        st.pyplot(eigen_fig)

# Reconstruct Image
if st.sidebar.button("Reconstruct Image"):
    st.session_state['reconstruct_button'] = True

if st.session_state['reconstruct_button']:
    uploaded_file = st.file_uploader("Upload an image to reconstruct", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('L')
        img_array = np.array(img)
        target_size = (st.session_state['img_width'], st.session_state['img_height'])
        img_resized = cv2.resize(img_array, target_size)
        img_flattened = img_resized.flatten().reshape(1, -1)

        img_pca = st.session_state['pca'].transform(img_flattened)
        img_reconstructed = st.session_state['pca'].inverse_transform(img_pca)

        reconstruct_fig = show_reconstructed_image(img_resized, img_reconstructed, target_size)
        st.pyplot(reconstruct_fig)
