# 🏦 Loan Status Prediction App

[![Streamlit App](https://img.shields.io/badge/Built%20With-Streamlit-%23FF4B4B?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/github/license/yourusername/loan-status-app)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)]()

A sleek, interactive Streamlit-based web application that predicts whether a loan application will be approved or rejected based on various customer details such as income, credit history, and education.

---

## 🚀 Live Demo
🌐 *Coming Soon* — Deploy this app easily on **Streamlit Cloud**, **Render**, or **Heroku**!

---

## 📌 Features

- 📊 **EDA (Exploratory Data Analysis):**
  Analyze dataset distributions, missing values, categorical & numerical variables.

- 🧠 **Model Training:**
  Impute missing values, apply scaling, feature selection (Chi²), and train models like RandomForest, AdaBoost, and SVM.

- 🧪 **Inference:**
  Input customer details and predict loan approval status in real-time.

- 📈 **Feature Selection:**
  Use Chi-Square test to select the most informative features.

- 💾 **Model Export:**
  Download trained ML models as `.pkl` files using Pickle.

---

## 🖼️ Screenshots

| Home Page | EDA | Model Training | Inference |
|----------|-----|----------------|-----------|
| ![Home](https://i.imgur.com/abcd123.png) | ![EDA](https://i.imgur.com/abcd456.png) | ![Train](https://i.imgur.com/abcd789.png) | ![Predict](https://i.imgur.com/abcd999.png) |

---

## 📁 Project Structure

loan-status-app/
├── app.py
├── sections/
│ ├── eda.py
│ ├── model_training.py
│ ├── inference.py
├── assets/
│ └── style.css
├── utils/
│ └── model_utils.py
├── README.md
└── requirements.txt

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/loan-status-app.git
cd loan-status-app
pip install -r requirements.txt
streamlit run app.py
