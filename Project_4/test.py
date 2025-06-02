import streamlit as st
from sections import EDA,Train,FeatureEngenieering
st.set_page_config(page_title="Loan Status Prediction", layout="wide")


st.sidebar.title("📁 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "🧠 Feature Engenieering",'Training', "🧾 Inference"])


if page == "🏠 Home":
    st.title("🏦 Loan Status Prediction App")
    st.markdown("#### A sleek ML-powered web app to predict loan approvals.")

    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🔍 About This Project
        This app uses Depp Learning Nueral Network to predict the price of the mobile based on various veatures

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
elif page == "🧠 Feature Engenieering":
    FeatureEngenieering.run()

elif page == "Training":
    Train.run()