# ---------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Employee Salary Regression Model Comparison",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("📊 Employee Salary Prediction & Model Comparison Dashboard")
st.write(
"""
This interactive machine learning dashboard compares multiple regression models
to predict **employee salary based on years of experience**.
"""
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Salary_Data.csv")
    return df

df = load_data()

# ---------------------------------------------------
# DATA OVERVIEW
# ---------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df)

# ---------------------------------------------------
# FEATURE / TARGET
# ---------------------------------------------------
X = df[['YearsExperience']]
y = df['Salary']

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# SCALING
# ---------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------
# MODELS
# ---------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

results = []
predictions_dict = {}

# ---------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------
for name, model in models.items():

    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    results.append({
        "Model": name,
        "R² Score": round(r2, 4),
        "MAE": round(mae, 2)
    })

    predictions_dict[name] = preds

results_df = pd.DataFrame(results)

# ---------------------------------------------------
# MODEL PERFORMANCE DASHBOARD
# ---------------------------------------------------
st.subheader("📈 Model Performance Comparison")

col1, col2 = st.columns(2)

with col1:
    fig_r2 = px.bar(
        results_df,
        x="Model",
        y="R² Score",
        title="Model R² Score Comparison",
        text="R² Score",
        color="Model"
    )

    st.plotly_chart(fig_r2, use_container_width=True)

with col2:
    fig_mae = px.bar(
        results_df,
        x="Model",
        y="MAE",
        title="Model MAE Comparison",
        text="MAE",
        color="Model"
    )

    st.plotly_chart(fig_mae, use_container_width=True)

# ---------------------------------------------------
# MODEL RANKING
# ---------------------------------------------------
st.subheader("🏆 Model Ranking")

ranking = results_df.sort_values(by="R² Score", ascending=False)
ranking.index = ranking.index + 1

st.dataframe(ranking)

# ---------------------------------------------------
# PREDICTION VS ACTUAL
# ---------------------------------------------------
st.subheader("📊 Prediction vs Actual Visualization")

selected_model = st.selectbox(
    "Select Model",
    list(predictions_dict.keys())
)

preds = predictions_dict[selected_model]

comparison_df = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": preds
})

fig_compare = px.scatter(
    comparison_df,
    x="Actual Salary",
    y="Predicted Salary",
    title=f"{selected_model} Prediction vs Actual",
    trendline="ols"
)

st.plotly_chart(fig_compare, use_container_width=True)

# ---------------------------------------------------
# SALARY PREDICTION TOOL
# ---------------------------------------------------
st.subheader("💰 Predict Salary")

years = st.slider(
    "Years of Experience",
    0.0,
    20.0,
    5.0
)

best_model_name = ranking.iloc[0]["Model"]
best_model = models[best_model_name]

best_model.fit(X_train_scaled, y_train)

input_scaled = scaler.transform([[years]])

prediction = best_model.predict(input_scaled)

st.success(
    f"Estimated Salary using **{best_model_name}**: ₹ {prediction[0]:,.2f}"
)

# ---------------------------------------------------
# DOWNLOAD REPORT
# ---------------------------------------------------
st.subheader("📥 Download Prediction Report")

report = comparison_df.copy()
report["Model Used"] = selected_model

csv = report.to_csv(index=False)

st.download_button(
    label="Download CSV Report",
    data=csv,
    file_name="salary_prediction_report.csv",
    mime="text/csv"
)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Scikit-learn & Plotly")
