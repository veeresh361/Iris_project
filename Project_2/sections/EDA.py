import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from config import DATA_LOAD_PATH

def draw_cat_graph(df,col_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='silver')
    gender_counts=df[col_name].value_counts()
    sns.countplot(data=df, x=col_name, ax=axes[0],hue=col_name, palette='Set2')
    # axes[0].set_xlabel(column_name.replace('_', ' ').title())
    axes[0].set_ylabel('Count')
    axes[1].pie(gender_counts.values,labels=gender_counts.index,
                autopct='%1.2f%%'
                ,shadow=True,
                colors=sns.color_palette("Set2"),
                startangle=140,
                explode=[0.05] * len(gender_counts))
    return fig

CONT_COLUMNS=['CoapplicantIncome','LoanAmount','Loan_Amount_Term']

def get_cont_graph(df,columns,graph_type='hist'):
    num_rows=1
    num_cols=3
    palette = sns.color_palette("dark", len(columns))
    fig, axs = plt.subplots(num_rows,num_cols, figsize=(14, 4.5 * num_rows), facecolor='silver')
    if graph_type=='HISTOGRAM':
        sns.histplot(data=df, x=CONT_COLUMNS[0], color=palette[0], kde=True, bins=round(np.sqrt(len(df))), ax=axs[0])
        sns.histplot(data=df, x=CONT_COLUMNS[1], color=palette[1], kde=True, bins=round(np.sqrt(len(df))), ax=axs[1])
        sns.histplot(data=df, x=CONT_COLUMNS[2], color=palette[2], kde=True, bins=round(np.sqrt(len(df))), ax=axs[2])
        return fig
    elif graph_type=="KDE":
        sns.kdeplot(data=df[CONT_COLUMNS[0]] , color=palette[0],fill=True,  ax=axs[0])
        sns.kdeplot(data=df[CONT_COLUMNS[1]] , color=palette[1],fill=True , ax=axs[1])
        sns.kdeplot(data=df[CONT_COLUMNS[2]], color=palette[2], fill=True, ax=axs[2])
        return fig

    elif graph_type=='BOX':
      sns.boxplot(data=df, y=CONT_COLUMNS[0], color=palette[0],ax=axs[0])
      sns.boxplot(data=df, y=CONT_COLUMNS[1], color=palette[1],ax=axs[1])
      sns.boxplot(data=df, y=CONT_COLUMNS[2], color=palette[2],ax=axs[2])
      return fig

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
        ("Show Data", "Missing Values", "Categorical Analysis", "Numerical Analysis")
    )

    # ------------------------------------
    # 1. Show Data
    # ------------------------------------
    if eda_option == "Show Data":
        st.subheader("🗂 Raw Dataset")
        rows = st.slider("Select number of rows to display", 5, len(st.session_state['data']), 10)
        st.dataframe(st.session_state['data'].head(rows))

    # ------------------------------------
    # 2. Missing Values
    # ------------------------------------
    elif eda_option == "Missing Values":
        st.subheader("🔍 Missing Values Overview")
        missing_df=st.session_state['data'].iloc[:,1:].isna().sum()
        missing_values_sum = st.session_state['data'].isna().iloc[:, 1:].sum()
        missing_df=pd.DataFrame(missing_df).T
        fig, ax = plt.subplots(figsize=(22, 5))
        sns.heatmap(missing_df,annot=True,fmt='0.0f',cmap=sns.dark_palette("gray",as_cmap=True),linecolor='silver',linewidth=1,ax=ax)
        plt.title('Count of Missing Values',fontsize=12,fontweight='bold')
        ax.set_xticklabels([label.replace('_', ' ').title() for label in missing_values_sum.index], fontsize=15, fontweight='bold')
        st.pyplot(fig)

    # ------------------------------------
    # 3. Categorical Column Analysis
    # ------------------------------------
    elif eda_option == "Categorical Analysis":
        st.subheader("📌 Categorical Feature Analysis")

        cat_cols = st.session_state['data'].select_dtypes(include="object").columns.tolist()
        if not cat_cols:
            st.warning("⚠️ No categorical columns found in the dataset.")
            return

        selected_cat = st.selectbox("Choose a categorical column", cat_cols)
        cat_figure=draw_cat_graph(st.session_state['data'],selected_cat)
        st.pyplot(cat_figure)

    # # ------------------------------------
    # # 4. Numerical Column Analysis
    # # ------------------------------------
    elif eda_option == "Numerical Analysis":
        st.subheader("📈 Numerical Feature Analysis")

        selected_num = st.selectbox("Choose a numerical column", ['KDE','HISTOGRAM','BOX'])

        cont_figure = get_cont_graph(st.session_state['data'],CONT_COLUMNS,selected_num)

        st.pyplot(cont_figure)
