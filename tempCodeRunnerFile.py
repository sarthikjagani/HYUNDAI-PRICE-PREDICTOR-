import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load data from a CSV file."""
    uploaded_file = st.file_uploader("Upload your Hyundai dataset (CSV file):", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

def preprocess_data(df):
    """Preprocess data for machine learning."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["year", "mileage", "tax(£)", "mpg", "engineSize"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["model", "transmission", "fuelType"]),
        ]
    )
    features = ["model", "year", "mileage", "transmission", "fuelType", "tax(£)", "mpg", "engineSize"]
    target = "price"
    return preprocessor, features, target

def train_model(df, preprocessor, features, target):
    """Train a Random Forest model."""
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        "R²": r2,
        "MAE": mae,
        "RMSE": rmse,
    }
    return model, metrics

def predict_price(model, input_data):
    """Predict car price based on user input."""
    prediction = model.predict(input_data)[0]
    return prediction

def show_visualizations(df):
    """Show dataset visualizations."""
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["price"], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Pairwise Relationships")
    pairplot_fig = sns.pairplot(df, vars=["year", "mileage", "price"])
    st.pyplot(pairplot_fig)

def main():
    st.title("Hyundai Car Price Prediction App")

    # Load data
    df = load_data()
    if df is not None:
        st.write("### Dataset Preview")
        st.write(df.head())

        # Preprocess data
        preprocessor, features, target = preprocess_data(df)

        # Train model
        model, metrics = train_model(df, preprocessor, features, target)

        st.write("### Model Performance")
        st.write(metrics)

        # Input for prediction
        st.write("### Predict Car Price")
        model_name = st.selectbox("Model", options=df["model"].unique())
        year = st.slider("Year", min_value=2000, max_value=2025, value=2010)
        mileage = st.slider("Mileage (km)", min_value=0, max_value=300000, value=50000)
        transmission = st.selectbox("Transmission", options=df["transmission"].unique())
        fuel_type = st.selectbox("Fuel Type", options=df["fuelType"].unique())
        tax = st.slider("Tax (£)", min_value=0, max_value=500, value=100)
        mpg = st.slider("MPG", min_value=0, max_value=100, value=50)
        engine_size = st.slider("Engine Size (L)", min_value=0.0, max_value=5.0, value=1.5)

        input_data = pd.DataFrame(
            {
                "model": [model_name],
                "year": [year],
                "mileage": [mileage],
                "transmission": [transmission],
                "fuelType": [fuel_type],
                "tax(£)": [tax],
                "mpg": [mpg],
                "engineSize": [engine_size],
            }
        )

        if st.button("Predict Price"):
            predicted_price = predict_price(model, input_data)
            st.write(f"### Predicted Price: £{predicted_price:,.2f}")

        # Visualizations
        st.write("### Data Visualizations")
        show_visualizations(df)

if __name__ == "__main__":
    main()
