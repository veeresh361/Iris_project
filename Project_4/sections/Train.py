import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
# from sections.utils import plot_chi2_scores
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler












def run():
    st.title("🧠 Model Training & Evaluation")
    st.subheader("🔍 Dataset Preview")
    st.dataframe(st.session_state['data'].head(5))
    st.subheader("2️⃣ Feature Scaling")
    scaler_option = st.radio("Choose Scaling Method", ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])

    if scaler_option != "None":
        scaler = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler()
        }[scaler_option]

        X = st.session_state['data'].drop('price_range', axis=1)
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        y = st.session_state['data']['price_range']
        st.success(f"✅ Features scaled using {scaler_option}")
        st.subheader("🔍 Scaled Dataset Preview")
        st.dataframe(X.head(5))