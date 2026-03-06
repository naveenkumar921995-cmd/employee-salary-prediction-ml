# -------------------------------------------------------
# Employee Salary Prediction - Regression Model Comparison
# Author: Naveen Kumar
# -------------------------------------------------------

# Import Libraries
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

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

dataset = pd.read_csv("data/emp_sal.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# -------------------------------------------------------
# Train Models
# -------------------------------------------------------

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

# -------------------------------------------------------
# Predictions
# -------------------------------------------------------

test_value = [[6.5]]

predictions = {
    "Linear Regression": lin_reg.predict(test_value)[0],
    "Polynomial Regression": poly_model.predict(poly.transform(test_value))[0],
    "SVR": svr.predict(test_value)[0],
    "KNN": knn.predict(test_value)[0],
    "Decision Tree": dt.predict(test_value)[0],
    "Random Forest": rf.predict(test_value)[0]
}

print("\nPrediction for Position Level = 6.5\n")

for model, value in predictions.items():
    print(f"{model}: {value:.2f}")

# -------------------------------------------------------
# Model Evaluation
# -------------------------------------------------------

evaluation = []

models = {
    "Linear Regression": lin_reg.predict(X),
    "Polynomial Regression": poly_model.predict(X_poly),
    "SVR": svr.predict(X),
    "KNN": knn.predict(X),
    "Decision Tree": dt.predict(X),
    "Random Forest": rf.predict(X)
}

for name, pred in models.items():

    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))

    evaluation.append([name, r2, mae, rmse])

results = pd.DataFrame(
    evaluation,
    columns=["Model", "R2 Score", "MAE", "RMSE"]
)

print("\nModel Performance Comparison\n")
print(results)

# -------------------------------------------------------
# Visualization
# -------------------------------------------------------

# Linear Regression Plot
plt.scatter(X, y)
plt.plot(X, lin_reg.predict(X))
plt.title("Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Polynomial Regression Plot
plt.scatter(X, y)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)

plt.plot(X_grid, poly_model.predict(poly.transform(X_grid)))

plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# -------------------------------------------------------
# Model Comparison Chart
# -------------------------------------------------------

plt.bar(results["Model"], results["R2 Score"])
plt.title("Model Performance Comparison (R² Score)")
plt.xticks(rotation=45)
plt.ylabel("R² Score")
plt.show()
