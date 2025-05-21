# 🧠 PCA Image Explorer

A visual tool to explore Principal Component Analysis (PCA) on images using a web-based interface built with **Streamlit**.

This application allows users to:

- 📷 Load and preview grayscale images
- 🧬 Apply PCA to extract and visualize eigenfaces
- 📊 Plot explained variance ratio
- 🔁 Reconstruct an uploaded image using PCA
- 🛠 Interactively configure settings via the sidebar

---

## 📁 Folder Structure
project/
│
├── app.py # Main Streamlit app
├── pca_utils.py # PCA processing utilities
├── plotting_utils.py # Visualization utilities
├── pca_config.py # Global constants (e.g., paths)
├── models/ # Directory for saving PCA models
├── images/ # Directory containing input images
├── requirements.txt # Python dependencies
└── README.md # You're here

---

## 🧰 Requirements

Install Python packages required to run the app:

```bash
pip install -r requirements.txt
