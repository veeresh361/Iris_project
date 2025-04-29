import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
from utils import scatter_plot, get_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from config import RESUTDICT


@st.cache_data
def load_data():
    data = load_iris()
    dat = pd.DataFrame(data.data, columns=data.feature_names)
    dat["target"] = data.target
    return dat


st.markdown(
    """
    <style>
    /* Background color for the entire page */
    .stApp {
        background-color:lightgreen;
    }
    /* Styling the title */
    h1 {
        color: #3366cc;
        font-size: 48px;
    }

    /* Styling the headers */
    h2 {
        color: #003366;
        font-size: 36px;
    }
    
    h3 {
        color: #336699;
        font-size: 30px;
    }

    /* Optional: Centering text */
    .stMarkdown {
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

iris_data = load_data()
# print(iris_data.head())
# exit()
st.sidebar.header("🌸 User Input Features")
st.sidebar.write("Adjust the features to classify the Iris flower:")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    float(iris_data["sepal length (cm)"].min()),
    float(iris_data["sepal length (cm)"].max()),
)
sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    float(iris_data["sepal width (cm)"].min()),
    float(iris_data["sepal width (cm)"].max()),
)
petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    float(iris_data["petal length (cm)"].min()),
    float(iris_data["petal length (cm)"].max()),
)
petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    float(iris_data["petal width (cm)"].min()),
    float(iris_data["petal width (cm)"].max()),
)


st.title("Iris Prediction App")
st.subheader("Predict the type of Iris flower based on measurement")
# st.dataframe(iris_data)
tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "EDA", "Model Training", "Prediction"])
with tab1:
    st.dataframe(iris_data)


with tab2:
    # st.write('EDA')
    x_axis = st.selectbox(
        "Select X axis", ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    y_axis = st.selectbox(
        "Select Y axis", ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    figre = scatter_plot(x_axis, y_axis)
    st.plotly_chart(figre)

with tab3:
    make_choice = st.selectbox(
        "Select the classifier",
        ("RanfomForest", "AdaBoost", "DecisionTreeClassifier", "LogisticRegression"),
    )
    if st.button("Train model"):
        selected_model = get_model(make_choice)
        x = iris_data.drop(["target"], axis=1)
        y = iris_data["target"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
        selected_model.fit(x_train, y_train)
        path = f"D:\\Projects\\Test\\models\\{make_choice}.pkl"
        joblib.dump(selected_model, path)
        st.session_state["trained_model"] = selected_model
        y_pred = selected_model.predict(x_test)
        score = accuracy_score(y_pred, y_test)
        st.success(f"Model choice {make_choice} Accuracy score is {score}")
        st.subheader("Confusion Matrix")
        matrix = confusion_matrix(y_test, y_pred)
        fig, axis = plt.subplots()
        sns.heatmap(matrix, annot=True, linewidth=0.5, ax=axis)
        st.pyplot(fig)

with tab4:
    st.write("Prediction")
    if st.button("Prediction"):

        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = st.session_state["trained_model"].predict(input_data)
        print(RESUTDICT[prediction[0]])
        st.write(RESUTDICT[prediction[0]])
