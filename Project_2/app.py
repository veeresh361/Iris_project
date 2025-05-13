
import streamlit as st
from sections import EDA,Train
st.set_page_config(page_title="Loan Status Prediction", layout="wide")

# ----------------------------------
# 🔹 Sidebar Navigation
# ----------------------------------
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "🧠 Model Training", "🧾 Inference"])

# ----------------------------------
# 🔹 Home Page
# ----------------------------------
if page == "🏠 Home":
    st.title("🏦 Loan Status Prediction App")
    st.markdown("#### A sleek ML-powered web app to predict loan approvals.")

    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🔍 About This Project
        This app uses machine learning to predict whether a loan application will be approved based on various customer attributes such as credit history, income, and more.

        **Features:**
        - 📊 Detailed Exploratory Data Analysis (EDA)
        - 🧠 Interactive Model Training
        - 🧾 Real-time Prediction Inference

        Built with ❤️ using [Streamlit](https://streamlit.io/), [Scikit-learn](https://scikit-learn.org/) & [Plotly](https://plotly.com/).

        ---
        """)
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/639/639365.png", width=200)

    st.success("Use the sidebar to navigate through the app 🚀")


elif page == "📊 EDA":
    EDA.run()
elif page == "🧠 Model Training":
    Train.run()

