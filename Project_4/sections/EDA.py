import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from config import DATA_LOAD_PATH


def count_graph(df,column_name):
    fig,axes=plt.subplots(1,3,figsize=(14,4),facecolor='silver')
    sns.histplot(data=df,x=column_name,kde=True,color=sns.set_palette("Set2"),ax=axes[0])
    sns.boxplot(data=df,x=column_name,color=sns.set_palette("Set2"),ax=axes[1])
    sns.kdeplot(data=df,x=column_name,color=sns.set_palette("Set2"),ax=axes[2])
    return fig

def run():
    st.title("📊 Exploratory Data Analysis")

    # Load dataset (replace with your own)
    @st.cache_data
    def load_data():
        return pd.read_csv(DATA_LOAD_PATH)
    

    st.session_state['data'] = load_data()
    st.session_state['data']=st.session_state['data'].iloc[:,1:]

    # Sub-navigation within EDA
    eda_option = st.sidebar.selectbox(
        "Select EDA Section",
        ("Show Data","Numerical Analysis")
    )
    if eda_option == "Show Data":
        st.subheader("🗂 Raw Dataset")
        rows = st.slider("Select number of rows to display", 5, len(st.session_state['data']), 10)
        st.dataframe(st.session_state['data'].head(rows))

    elif eda_option == "Numerical Analysis":
        st.subheader("📈 Numerical Feature Analysis")
        CONT_COLS=[col for col in st.session_state['data'].columns if st.session_state['data'][col].dtype!='object']
        selected_column=st.selectbox('Choose the column name',CONT_COLS)
        cont_figure=count_graph(st.session_state['data'],selected_column)
        st.pyplot(cont_figure)