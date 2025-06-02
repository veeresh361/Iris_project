import pickle
import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif

def plot_chi2_scores(anova_df, title="Feature Importance Anova", palette="Set2"):
    """
    Plots Chi² scores for features using a horizontal bar plot.

    Parameters:
        chi2_df (pd.DataFrame): DataFrame with 'Feature' and 'Chi2 Score' columns.
        title (str): Title of the plot.
        palette (str): Seaborn color palette.
    """
    # Sort for better visualization
    chi2_df_sorted = anova_df.sort_values(by="F-Score", ascending=False)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="whitesmoke")

    sns.barplot(
        x="F-Score",
        y="Feature",
        data=chi2_df_sorted,
        hue='Feature',
        ax=ax,
        palette=palette
    )

    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Chi² Score")
    ax.set_ylabel("Feature")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    return fig


def run():
    if 'features_importance' not in st.session_state:
      st.session_state['features_importance'] = None

    if 'anova_results' not in st.session_state:
      st.session_state['anova_results'] = None

    if 'show_graph' not in st.session_state:
      st.session_state['show_graph'] = None

    if 'train_data' not in st.session_state:
      st.session_state['train_data'] = None

    if 'target' not in st.session_state:
      st.session_state['target'] = None
    
    st.title("🧠 Model Training & Evaluation")
    st.subheader("🔍 Dataset Preview")
    st.dataframe(st.session_state['data'].head(5))

    st.subheader("✨ Feature Selection")
    st.markdown("""
    Use ANOVA statistical test to select the most relevant features based on their relationship with the target variable.
    """)
    if st.button('Perform Chi Square Test'):
        st.session_state['features_importance']=True
        st.session_state['show_graph']=True
        selector = SelectKBest(score_func=f_classif)
        X, y = st.session_state['data'].drop(columns=['price_range']),st.session_state['data']['price_range']
        selector.fit(X, y)
        st.session_state['anova_results'] = pd.DataFrame({
        'Feature': X.columns,
        'F-Score': selector.scores_,
        'p-Value': selector.pvalues_}).sort_values(by='F-Score', ascending=False)
        st.session_state['anova_plot'] = plot_chi2_scores(st.session_state['anova_results'])
    if st.session_state['show_graph']==True:
       st.pyplot(st.session_state['anova_plot'])

    if st.session_state['features_importance']==True:
          top_k = st.slider("🎯 Select Top K Features", min_value=1, max_value=len(st.session_state['anova_results']), value=6)
          top_features = st.session_state['anova_results'].head(top_k)['Feature'].tolist()
          st.info(f"Selected Features Based on Anova: {', '.join(top_features)}")
          st.dataframe(st.session_state['data'][top_features].head(5))
          st.session_state['selected_features']=top_features

    st.subheader("2️⃣ Feature Scaling")
    scaler_option = st.radio("Choose Scaling Method", ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])

    if scaler_option != "None":
        scaler = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler()
        }[scaler_option]

        X = st.session_state['data'].drop('price_range', axis=1)
        X = pd.DataFrame(scaler.fit_transform(X[st.session_state['selected_features']]), columns=st.session_state['selected_features'])
        y = st.session_state['data']['price_range']
        st.success(f"✅ Features scaled using {scaler_option}")
        st.subheader("🔍 Scaled Dataset Preview")
        st.dataframe(X.head(5))
        st.session_state['train_data']= X
        st.session_state['target']=y