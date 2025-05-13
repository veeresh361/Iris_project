import plotly.express as px
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def scatter_plot(x_axis,y_axis):
    df = px.data.iris()
    fig = px.scatter(df, x=x_axis, y=y_axis, color="species",
                     size='petal_length', hover_data=['petal_width'])
    return fig

def get_model(choice):
    if choice=="RanfomForest":
        model=RandomForestClassifier()
    elif choice=="AdaBoost":
        model=AdaBoostClassifier()
    elif choice=="DecisionTreeClassifier":
        model=DecisionTreeClassifier()
    else:
        model=LogisticRegression()
    return model
