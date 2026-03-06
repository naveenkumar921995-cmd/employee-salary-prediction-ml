# ---------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------
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

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Employee Salary ML Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Salary Prediction Dashboard")

st.write(
"""
This interactive Machine Learning dashboard compares multiple regression models  
to determine the **best algorithm for salary prediction**.
"""
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("emp_sal.csv")
    return df

df = load_data()

# ---------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df)

# ---------------------------------------------------
# FEATURES
# ---------------------------------------------------
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

# ---------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Polynomial Regression
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X, y)

# KNN
knn_reg = KNeighborsRegressor(n_neighbors=3, weights='distance')
knn_reg.fit(X, y)

# Decision Tree
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X, y)

# Random Forest
rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
rf_reg.fit(X, y)

# ---------------------------------------------------
# MODEL PREDICTIONS (test value)
# ---------------------------------------------------
test_value = np.array([[6.5]])

results = {
    "Linear Regression": lin_reg.predict(test_value)[0],
    "Polynomial Regression": poly_reg.predict(poly.transform(test_value))[0],
    "SVR": svr_reg.predict(test_value)[0],
    "KNN": knn_reg.predict(test_value)[0],
    "Decision Tree": dt_reg.predict(test_value)[0],
    "Random Forest": rf_reg.predict(test_value)[0]
}

results_df = pd.DataFrame(
    list(results.items()),
    columns=["Model", "Predicted Salary"]
)

# ---------------------------------------------------
# MODEL COMPARISON CHART
# ---------------------------------------------------
st.subheader("📊 Model Prediction Comparison")

fig = px.bar(
    results_df,
    x="Model",
    y="Predicted Salary",
    color="Model",
    text="Predicted Salary",
    title="Salary Prediction Comparison (Position Level = 6.5)"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# REGRESSION VISUALIZATION
# ---------------------------------------------------
st.subheader("Regression Curve Visualization")

# FIXED np.arange ERROR
X_grid = np.arange(start=X.min(), stop=X.max(), step=0.1)
X_grid = X_grid.reshape(-1, 1)

model_option = st.selectbox(
    "Select Model",
    [
        "Linear Regression",
        "Polynomial Regression",
        "SVR",
        "KNN",
        "Decision Tree",
        "Random Forest"
    ]
)

if model_option == "Linear Regression":
    y_pred = lin_reg.predict(X_grid)

elif model_option == "Polynomial Regression":
    y_pred = poly_reg.predict(poly.transform(X_grid))

elif model_option == "SVR":
    y_pred = svr_reg.predict(X_grid)

elif model_option == "KNN":
    y_pred = knn_reg.predict(X_grid)

elif model_option == "Decision Tree":
    y_pred = dt_reg.predict(X_grid)

else:
    y_pred = rf_reg.predict(X_grid)

plot_df = pd.DataFrame({
    "Position Level": X_grid.flatten(),
    "Predicted Salary": y_pred
})

fig2 = px.line(
    plot_df,
    x="Position Level",
    y="Predicted Salary",
    title=f"{model_option} Regression Curve"
)

fig2.add_scatter(
    x=X.flatten(),
    y=y,
    mode="markers",
    name="Actual Data"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------
# MODEL RANKING
# ---------------------------------------------------
st.subheader("🏆 Model Ranking")

ranking = results_df.sort_values(
    by="Predicted Salary",
    ascending=False
)

ranking.index = ranking.index + 1

st.dataframe(ranking)

# ---------------------------------------------------
# SALARY PREDICTION TOOL
# ---------------------------------------------------
st.subheader("💰 Salary Prediction Tool")

level = st.slider(
    "Select Position Level",
    1.0,
    10.0,
    5.0
)

prediction = poly_reg.predict(poly.transform([[level]]))

st.success(
    f"Estimated Salary: ₹ {prediction[0]:,.2f}"
)

# ---------------------------------------------------
# DOWNLOAD REPORT
# ---------------------------------------------------
st.subheader("📥 Download Prediction Report")

csv = results_df.to_csv(index=False)

st.download_button(
    label="Download CSV Report",
    data=csv,
    file_name="salary_predictions.csv",
    mime="text/csv"
)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Python • Streamlit • Scikit-Learn • Plotly")
