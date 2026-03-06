# ==========================================
# IMPORT LIBRARIES
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI Salary Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("emp_sal.csv")
    return df

df = load_data()

st.title("📊 Employee Salary Prediction & Model Comparison")

st.write("Machine Learning dashboard comparing multiple regression models.")

st.dataframe(df)

# ==========================================
# FEATURES
# ==========================================
X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

# ==========================================
# TRAIN MODELS
# ==========================================

# Linear
lin_model = LinearRegression()
lin_model.fit(X,y)

# Polynomial
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly,y)

# SVR
svr_model = SVR(kernel="rbf")
svr_model.fit(X,y)

# KNN
knn_model = KNeighborsRegressor(n_neighbors=3,weights="distance")
knn_model.fit(X,y)

# Decision Tree
dt_model = DecisionTreeRegressor(random_state=0)
dt_model.fit(X,y)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=200,random_state=0)
rf_model.fit(X,y)

# ==========================================
# MODEL ACCURACY
# ==========================================
lin_r2 = r2_score(y, lin_model.predict(X))
poly_r2 = r2_score(y, poly_model.predict(X_poly))
svr_r2 = r2_score(y, svr_model.predict(X))
knn_r2 = r2_score(y, knn_model.predict(X))
dt_r2 = r2_score(y, dt_model.predict(X))
rf_r2 = r2_score(y, rf_model.predict(X))

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Prediction Controls")

level = st.sidebar.slider(
    "Employee Position Level",
    float(X.min()),
    float(X.max()),
    6.5
)

model_select = st.sidebar.selectbox(
    "Select Model",
    [
        "Auto Best Model",
        "Linear Regression",
        "Polynomial Regression",
        "SVR",
        "KNN",
        "Decision Tree",
        "Random Forest"
    ]
)

# ==========================================
# MODEL SCORES
# ==========================================
scores = {
"Linear Regression": lin_r2,
"Polynomial Regression": poly_r2,
"SVR": svr_r2,
"KNN": knn_r2,
"Decision Tree": dt_r2,
"Random Forest": rf_r2
}

# Auto best model
if model_select == "Auto Best Model":
    model_select = max(scores, key=scores.get)

# ==========================================
# PREDICTION
# ==========================================
if model_select == "Linear Regression":
    prediction = lin_model.predict([[level]])

elif model_select == "Polynomial Regression":
    prediction = poly_model.predict(poly.transform([[level]]))

elif model_select == "SVR":
    prediction = svr_model.predict([[level]])

elif model_select == "KNN":
    prediction = knn_model.predict([[level]])

elif model_select == "Decision Tree":
    prediction = dt_model.predict([[level]])

else:
    prediction = rf_model.predict([[level]])

# ==========================================
# KPI CARDS
# ==========================================
col1,col2,col3,col4 = st.columns(4)

col1.metric("Linear R²",round(lin_r2,3))
col2.metric("Polynomial R²",round(poly_r2,3))
col3.metric("Best Model",model_select)
col4.metric("Predicted Salary",f"${prediction[0]:,.0f}")

# ==========================================
# MODEL LEADERBOARD
# ==========================================
st.subheader("🏆 Model Accuracy Leaderboard")

leaderboard = pd.DataFrame({
"Model":scores.keys(),
"R² Score":scores.values()
}).sort_values("R² Score",ascending=False)

st.dataframe(leaderboard,use_container_width=True)

# ==========================================
# MODEL COMPARISON CHART
# ==========================================
fig = px.bar(
leaderboard,
x="Model",
y="R² Score",
color="Model",
title="Regression Model Accuracy Comparison"
)

st.plotly_chart(fig,use_container_width=True)

# ==========================================
# SALARY TREND
# ==========================================
st.subheader("📈 Polynomial Regression Trend")

X_grid = np.arange(X.min(),X.max(),0.1).reshape(-1,1)

fig2 = px.scatter(
x=X.flatten(),
y=y,
labels={"x":"Position Level","y":"Salary"}
)

fig2.add_scatter(
x=X_grid.flatten(),
y=poly_model.predict(poly.transform(X_grid)),
mode="lines",
name="Polynomial Regression"
)

st.plotly_chart(fig2,use_container_width=True)

# ==========================================
# PREDICTION COMPARISON
# ==========================================
st.subheader("📊 Model Prediction Comparison")

comparison = pd.DataFrame({
"Model":[
"Linear",
"Polynomial",
"SVR",
"KNN",
"Decision Tree",
"Random Forest"
],

"Prediction":[
lin_model.predict([[level]])[0],
poly_model.predict(poly.transform([[level]]))[0],
svr_model.predict([[level]])[0],
knn_model.predict([[level]])[0],
dt_model.predict([[level]])[0],
rf_model.predict([[level]])[0]
]
})

fig3 = px.bar(
comparison,
x="Model",
y="Prediction",
color="Model",
title="Salary Prediction by Different Models"
)

st.plotly_chart(fig3,use_container_width=True)
