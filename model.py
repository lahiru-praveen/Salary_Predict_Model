import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
data = pd.read_csv('Salary Data.csv')

# Split the data
X = data[['YearsExperience']]  # Features
y = data['Salary']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Model evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit Web App
st.title("Predict Salary")
st.write("This app predicts the salary based on years of experience using a simple linear regression model.")

# Sidebar for user input
st.sidebar.header("Input Parameters")
years_experience = st.sidebar.slider("Years of Experience", 0.0, 20.0, 5.0)

# Predict salary
user_input_df = pd.DataFrame({'YearsExperience': [years_experience]})
predicted_salary = model.predict(user_input_df)[0]

# Display the predicted salary
st.write(f"### Predicted Salary: ${predicted_salary:.2f}")

# Tabs for additional details
tab1, tab2, tab3 = st.tabs(["Plot", "Model Details", "Dataset Details"])

with tab1:
    st.write("### Actual vs Predicted Salary Plot")

    fig, ax = plt.subplots()
    ax.scatter(X_train, y_train, color='blue', label='Actual Salary (Training)')
    ax.scatter(X_test, y_test, color='red', label='Actual Salary (Test)')
    ax.plot(X_test, y_pred, color='green', linewidth=2, label='Predicted Salary')
    ax.set_xlabel('Years of Experience')
    ax.set_ylabel('Salary')
    ax.set_title('Actual vs Predicted Salary')
    ax.legend()

    st.pyplot(fig)

with tab2:
    st.write("### Model Performance Metrics")
    st.write(f"**Mean Squared Error (MSE):** {mse}")
    st.write(f"**R-squared:** {r2}")

    st.write("### Model Coefficients")
    st.write(f"**Intercept:** {model.intercept_}")
    st.write(f"**Coefficient (Slope):** {model.coef_[0]}")

with tab3:
    st.write("### Existing Dataset Details")
    st.dataframe(data)
    st.write(f"Number of Records: {data.shape[0]}")
    st.write(f"Number of Features: {data.shape[1]}")
