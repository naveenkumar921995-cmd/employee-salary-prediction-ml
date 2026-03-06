# -----------------------------------------------------------
# Employee Salary Prediction using Multiple Regression Models
# Author: Naveen Kumar
# -----------------------------------------------------------

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML Libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Evaluation Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------------------------------------------
# Load Dataset
# -----------------------------------------------------------

dataset = pd.read_csv("data/emp_sal.csv")

# Feature and Target
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# -----------------------------------------------------------
# Linear Regression Model
# -----------------------------------------------------------

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# -----------------------------------------------------------
# Polynomial Regression Model
# -----------------------------------------------------------

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# -----------------------------------------------------------
# Support Vector Regression Model
# -----------------------------------------------------------

svr = SVR(kernel="rbf")
svr.fit(X, y)

# -----------------------------------------------------------
# KNN Regression Model
# -----------------------------------------------------------

knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
knn.fit(X, y)

# -----------------------------------------------------------
# Decision Tree Regression
# -----------------------------------------------------------

dt = DecisionTreeRegressor(random_state=0)
dt.fit(X, y)

# -----------------------------------------------------------
# Random Forest Regression
# -----------------------------------------------------------

rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X, y)

# -----------------------------------------------------------
# Prediction Example
# -----------------------------------------------------------

test_value = [[6.5]]

lin_pred = lin_reg.predict(test_value)
poly_pred = poly_model.predict(poly.transform(test_value))
svr_pred = svr.predict(test_value)
knn_pred = knn.predict(test_value)
dt_pred = dt.predict(test_value)
rf_pred = rf.predict(test_value)

print("\nPrediction for Position Level = 6.5\n")

print("Linear Regression:", lin_pred)
print("Polynomial Regression:", poly_pred)
print("SVR:", svr_pred)
print("KNN:", knn_pred)
print("Decision Tree:", dt_pred)
print("Random Forest:", rf_pred)

# -----------------------------------------------------------
# Model Evaluation
# -----------------------------------------------------------

models = {
    "Linear Regression": lin_reg.predict(X),
    "Polynomial Regression": poly_model.predict(X_poly),
    "SVR": svr.predict(X),
    "KNN": knn.predict(X),
    "Decision Tree": dt.predict(X),
    "Random Forest": rf.predict(X)
}

print("\nModel Evaluation Metrics\n")

for name, prediction in models.items():

    r2 = r2_score(y, prediction)
    mae = mean_absolute_error(y, prediction)
    rmse = np.sqrt(mean_squared_error(y, prediction))

    print(f"{name}")
    print("R2 Score:", r2)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("----------------------------")

# -----------------------------------------------------------
# Visualization
# -----------------------------------------------------------

plt.scatter(X, y, color="red")
plt.plot(X, lin_reg.predict(X), color="blue")

plt.title("Linear Regression Fit")
plt.xlabel("Position Level")
plt.ylabel("Salary")

plt.show()
