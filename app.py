# ---------------------------------------------
# Employee Salary Prediction - ML Model Comparison
# ---------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------
# Page Title
# ---------------------------------------------

st.title("Employee Salary Prediction")
st.subheader("Regression Model Comparison Project")

# ---------------------------------------------
# Load Dataset
# ---------------------------------------------

dataset = pd.read_csv("data/emp_sal.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

st.write("Dataset Preview")
st.dataframe(dataset)

# ---------------------------------------------
# Train Models
# ---------------------------------------------

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Polynomial Regression
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# SVR
svr = SVR(kernel='rbf')
svr.fit(X, y)

# KNN
knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
knn.fit(X, y)

# Decision Tree
dt = DecisionTreeRegressor(random_state=0)
dt.fit(X, y)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X, y)

# ---------------------------------------------
# Prediction Section
# ---------------------------------------------

st.header("Salary Prediction")

level = st.slider("Select Position Level", 1.0, 10.0, 5.0)

test_value = [[level]]

predictions = {
    "Linear Regression": lin_reg.predict(test_value)[0],
    "Polynomial Regression": poly_model.predict(poly.transform(test_value))[0],
    "SVR": svr.predict(test_value)[0],
    "KNN": knn.predict(test_value)[0],
    "Decision Tree": dt.predict(test_value)[0],
    "Random Forest": rf.predict(test_value)[0]
}

pred_df = pd.DataFrame(
    predictions.items(),
    columns=["Model", "Predicted Salary"]
)

st.subheader("Prediction Comparison")
st.dataframe(pred_df)

# ---------------------------------------------
# Model Evaluation
# ---------------------------------------------

models = {
    "Linear Regression": lin_reg.predict(X),
    "Polynomial Regression": poly_model.predict(X_poly),
    "SVR": svr.predict(X),
    "KNN": knn.predict(X),
    "Decision Tree": dt.predict(X),
    "Random Forest": rf.predict(X)
}

evaluation = []

for name, pred in models.items():

    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))

    evaluation.append([name, r2, mae, rmse])

results = pd.DataFrame(
    evaluation,
    columns=["Model", "R2 Score", "MAE", "RMSE"]
)

st.subheader("Model Performance")
st.dataframe(results)

# ---------------------------------------------
# Linear Regression Chart
# ---------------------------------------------

st.subheader("Linear Regression Chart")

fig1, ax1 = plt.subplots()

ax1.scatter(X, y)
ax1.plot(X, lin_reg.predict(X))

ax1.set_xlabel("Position Level")
ax1.set_ylabel("Salary")
ax1.set_title("Linear Regression")

st.pyplot(fig1)

# ---------------------------------------------
# Polynomial Regression Chart
# ---------------------------------------------

st.subheader("Polynomial Regression Chart")

X_grid = np.arange(X.min(), X.max(), 0.1)
X_grid = X_grid.reshape(-1, 1)

fig2, ax2 = plt.subplots()

ax2.scatter(X, y)
ax2.plot(X_grid, poly_model.predict(poly.transform(X_grid)))

ax2.set_xlabel("Position Level")
ax2.set_ylabel("Salary")
ax2.set_title("Polynomial Regression")

st.pyplot(fig2)

# ---------------------------------------------
# Model Comparison Chart
# ---------------------------------------------

st.subheader("Model Comparison (R² Score)")

fig3, ax3 = plt.subplots()

ax3.bar(results["Model"], results["R2 Score"])

ax3.set_xlabel("Model")
ax3.set_ylabel("R2 Score")
ax3.set_title("Regression Model Comparison")

plt.xticks(rotation=45)

st.pyplot(fig3)

# ---------------------------------------------
# Best Model
# ---------------------------------------------

best_model = results.sort_values("R2 Score", ascending=False).iloc[0]

st.success(
    f"Best Model: {best_model['Model']} with R² Score = {best_model['R2 Score']:.3f}"
)
