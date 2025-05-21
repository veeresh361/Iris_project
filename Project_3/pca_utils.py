import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
import joblib
import streamlit as st

@st.cache_data(show_spinner=False)
def load_images_from_folder(folder_path, image_size=(64, 64)):
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img.flatten())
    return np.array(images), image_size

def apply_pca(X):
    pca = PCA(whiten=True)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

def save_pca_model(pca, filename='models/pca_model.pkl'):
    joblib.dump(pca, filename)

def load_pca_model(filename='models/pca_model.pkl'):
    return joblib.load(filename)

def reconstruct_image(pca, X_pca):
    return pca.inverse_transform(X_pca)
