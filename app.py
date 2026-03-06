# =================================================
# IMPORT LIBRARIES
# =================================================
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

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Employee Salary Model Comparison",
    layout="wide"
)

st.title("📊 Employee Salary Prediction Dashboard")

# =================================================
# LOAD DATA
# =================================================
df = pd.read_csv("emp_sal.csv")

st.subheader("Dataset")
st.dataframe(df)

# =================================================
# FEATURES
# =================================================
X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

# =================================================
# LINEAR REGRESSION
# =================================================
lin_model = LinearRegression()
lin_model.fit(X,y)

# =================================================
# POLYNOMIAL REGRESSION
# =================================================
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly,y)

# =================================================
# SVR
# =================================================
svr_model = SVR(kernel="rbf")
svr_model.fit(X,y)

# =================================================
# KNN
# =================================================
knn_model = KNeighborsRegressor(n_neighbors=3, weights="distance")
knn_model.fit(X,y)

# =================================================
# DECISION TREE
# =================================================
dt_model = DecisionTreeRegressor(random_state=0)
dt_model.fit(X,y)

# =================================================
# RANDOM FOREST
# =================================================
rf_model = RandomForestRegressor(n_estimators=200,random_state=0)
rf_model.fit(X,y)

# =================================================
# SIDEBAR INPUT
# =================================================
st.sidebar.header("Prediction Input")

level = st.sidebar.slider(
"Position Level",
1.0,
10.0,
6.5
)

# =================================================
# PREDICTIONS
# =================================================
lin_pred = lin_model.predict([[level]])[0]
poly_pred = poly_model.predict(poly.transform([[level]]))[0]
svr_pred = svr_model.predict([[level]])[0]
knn_pred = knn_model.predict([[level]])[0]
dt_pred = dt_model.predict([[level]])[0]
rf_pred = rf_model.predict([[level]])[0]

# =================================================
# RESULTS TABLE
# =================================================
results = pd.DataFrame({

"Model":[
"Linear Regression",
"Polynomial Regression",
"SVR",
"KNN",
"Decision Tree",
"Random Forest"
],

"Predicted Salary":[
lin_pred,
poly_pred,
svr_pred,
knn_pred,
dt_pred,
rf_pred
]

})

st.subheader("💰 Model Predictions")

st.dataframe(results)

# =================================================
# LEADERBOARD
# =================================================
leaderboard = results.sort_values(
"Predicted Salary",
ascending=False
)

st.subheader("🏆 Model Leaderboard")

st.dataframe(leaderboard)

# =================================================
# COMPARISON CHART
# =================================================
fig = px.bar(
results,
x="Model",
y="Predicted Salary",
color="Model",
title="Regression Model Salary Prediction Comparison"
)

st.plotly_chart(fig,use_container_width=True)

# =================================================
# REGRESSION CURVE
# =================================================
st.subheader("📈 Polynomial Regression Curve")

X_grid = np.arange(min(X),max(X),0.1).reshape(-1,1)

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
