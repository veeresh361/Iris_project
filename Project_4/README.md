# 🏦 Loan Status Prediction App

This project focuses on building and deploying a machine learning application that predicts whether a **Mobile Price Prediction App** based on ram,no of sims,wifi  etc.

Built using **Streamlit**, this app provides an end-to-end solution from data analysis to model training and real-time predictions.

---

## 📂 Project Structure

- `app.py` — Main Streamlit application file
- `assets/` — Optional folder for CSS or images
- `README.md` — Project documentation (this file)
- `requirements.txt` — List of required Python libraries
- `models/` — Folder to save/export trained ML models (optional)

---

## 📊 Dataset

The dataset used is the **Mobile Price Prediction** from [Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification).

It contains information like:

- No of sims
- ram size
- wifi
- Touch Screen 
- int_memory
- mobile_price (Target Variable)

---

## ⚙️ Technologies Used

- Python 3.9+
- Streamlit (Web UI)
- Pandas & NumPy (Data Handling)
- Seaborn & Matplotlib (Visualization)
- Scikit-learn (ML & Preprocessing)
- Pytorch

---

## 🚀 Project Workflow

1. Upload or load dataset
2. Perform Exploratory Data Analysis (EDA)
    - View raw data
    - Handle missing values
    - Analyze categorical & numerical features
3. Preprocess data
    - Impute missing values
    - Encode categorical variables
    - Feature scaling
4. Feature Selection
    - Apply **Anova Test** to pick top-k important features
5. Train DL models (user-selectable)
    - ANN
6. Evaluate model
    - Accuracy
    - Classification report
7. Save trained model using pytorch
8. Provide a download option for the trained `.pth` model

---

## 📈 Results

The model's performance may vary based on the selected features and preprocessing settings.
Typical accuracy ranges from **75% to 85%** depending on model and tuning.

---

## 💻 How to Run Locally

1. Clone this repository
    ```bash
    git clone https://github.com/veeresh361/Iris_project.git
    cd Project_2
    ```

2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

3. Launch the Streamlit app
    ```bash
    streamlit run app.py
    ```

---

## ✨ Features

- Sidebar navigation for multiple sections
- Interactive EDA with charts & value counts
- Missing value imputation (mean, median, mode)
- Feature scaling options (StandardScaler, MinMax, Robust)
- Feature selection via Anova  test
- Download trained model with one click

---

## 🎯 Future Enhancements

- Add cross-validation for better evaluation
- Hyperparameter tuning (GridSearchCV)
- Deploy on Streamlit Cloud with permanent link
- Add user authentication (admin mode)

---

## 🤝 Acknowledgements

- [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

## 👨‍💻 Author

**Your Name**
📧 your.email@example.com
🌐 [LinkedIn](https://linkedin.com/in/your-profile)
🐙 [GitHub](https://github.com/yourusername)

---

⭐️ _If you found this project helpful, give it a star!_
