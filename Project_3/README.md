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
```text
project/
├── app.py # Main Streamlit app
├── pca_utils.py # PCA processing utilities
├── plotting_utils.py # Visualization utilities
├── pca_config.py # Global constants (e.g., paths)
├── models/ # Directory for saving PCA models
├── images/ # Directory containing input images
├── requirements.txt # Python dependencies
└── README.md # You're here

---

## ⚙️ Technologies Used

- Python 3.9+
- Streamlit (Web UI)
- Pandas & NumPy (Data Handling)
- Seaborn & Matplotlib (Visualization)
- Scikit-learn (ML & Preprocessing)

---

## 💻 How to Run Locally

1. Clone this repository
    ```bash
    git clone https://github.com/veeresh361/Iris_project.git
    cd Project_3
    ```

2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

3. Launch the Streamlit app
    ```bash
    streamlit run app.py
