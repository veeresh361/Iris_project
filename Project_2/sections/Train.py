import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sections.utils import plot_chi2_scores
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from config import MODEL_SAVE_PATH



def fill_na_mode_mean(df,fill_type):
    df_filled = df.copy()

    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # Fill with mode (most frequent)
            mode_value = df[col].mode()[0]
            df_filled[col] = df[col].fillna(mode_value)
        else:
            # Fill with mean
            if fill_type=='Mean':
              mean_value = df[col].mean()
              df_filled[col] = df[col].fillna(mean_value)
            elif fill_type=='Mode':
              mode_value = df[col].mode()
              df_filled[col] = df[col].fillna(mode_value)
            else:
              median_value = df[col].median()
              df_filled[col] = df[col].fillna(median_value)


    return df_filled

def get_model(name):
        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "SVM": SVC()
        }
        return models[name]

def run():
    if 'features_importance' not in st.session_state:
      st.session_state['features_importance'] = None

    if 'chai_results' not in st.session_state:
      st.session_state['chai_results'] = None
    st.title("🧠 Model Training & Evaluation")
    st.subheader("🔍 Dataset Preview")
    st.dataframe(st.session_state['data'].head(5))

    st.subheader("1️⃣ Missing Value Handling")
    fill_method = st.radio("Choose imputation method for numeric features", ["Mean", "Median", "Mode"])
    st.session_state['filled_data']=fill_na_mode_mean(st.session_state['data'],fill_method)
    df_encoded = pd.get_dummies(st.session_state['filled_data'], drop_first=True)
    st.subheader("2️⃣ Feature Scaling")
    scaler_option = st.radio("Choose Scaling Method", ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])

    if scaler_option != "None":
        scaler = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler()
        }[scaler_option]
        target_col = st.selectbox("Select Target Column", df_encoded.columns)
        X = df_encoded.drop(target_col, axis=1)
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        y = df_encoded[target_col]
        st.success(f"✅ Features scaled using {scaler_option}")
        st.subheader("🔍 Scaled Dataset Preview")
        st.dataframe(X.head(5))
        st.subheader("✨ Feature Selection")
        st.markdown("""
        Use Chi-Square (χ²) statistical test to select the most relevant features based on their relationship with the target variable.
        """)
        if st.button('Perform Chi Square Test'):
          st.session_state['features_importance']=True
          chi_scores, p_values = chi2(X, y)
          st.session_state['chai_results'] = pd.DataFrame({
          'Feature': X.columns,
          'Chi2 Score': chi_scores,
          'p-value': p_values
            }).sort_values(by='Chi2 Score', ascending=False)
          feature_importance_graph=plot_chi2_scores(st.session_state['chai_results'])
          st.pyplot(feature_importance_graph)
        if st.session_state['features_importance']==True:
          top_k = st.slider("🎯 Select Top K Features", min_value=1, max_value=len(st.session_state['chai_results']), value=6)
          top_features = st.session_state['chai_results'].head(top_k)['Feature'].tolist()
          st.info(f"Selected Features Based on Chi²: {', '.join(top_features)}")
          st.dataframe(X[top_features].head(5))
          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
          model_text=st.selectbox('Select Model',['RandomForest','Decision Tree','AdaBoost','Gradient Boosting'])
          if st.button('Train Model'):
            model = get_model(model_text)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            st.success(f"✅ Model trained: {model_text}")
            st.metric(label="Accuracy", value=f"{acc:.2%}")
            st.text("📄 Classification Report:")
            st.code(report)

            with open(os.path.join(MODEL_SAVE_PATH,f"{model_text}.pkl"), 'wb') as f:
              pickle.dump(model, f)
            st.success('Model Trained and saved.')





