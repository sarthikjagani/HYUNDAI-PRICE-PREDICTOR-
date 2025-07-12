import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('hyundi.csv')

data = load_data()

# Initialize session state for chat and preferences
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {}

# Rasa chatbot integration
RASA_URL = "http://localhost:5005/webhooks/rest/webhook"

def send_to_rasa(message):
    """Send a message to the Rasa server and return the response."""
    payload = {"sender": "user", "message": message}
    try:
        response = requests.post(RASA_URL, json=payload)
        if response.status_code == 200:
            return [resp.get("text", "") for resp in response.json()]
        else:
            return ["Error communicating with Rasa server."]
    except Exception as e:
        return [f"Error: {e}"]

# Helper functions

def preprocess_data(df):
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

    y_pred = model.predict(X_test)
    metrics = {
        "R²": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    return model, metrics

def filter_cars(data, preferences):
    filtered_data = data
    if "budget" in preferences:
        min_budget, max_budget = preferences["budget"]
        filtered_data = filtered_data[(filtered_data["price"] >= min_budget) & (filtered_data["price"] <= max_budget)]
    if "fuelType" in preferences:
        filtered_data = filtered_data[filtered_data["fuelType"] == preferences["fuelType"]]
    if "mileage" in preferences:
        max_mileage = preferences["mileage"]
        filtered_data = filtered_data[filtered_data["mileage"] <= max_mileage]
    if "transmission" in preferences:
        filtered_data = filtered_data[filtered_data["transmission"] == preferences["transmission"]]
    return filtered_data

def compare_models(data, model1, model2):
    data["model_normalized"] = data["model"].str.strip().str.lower()
    cars1 = data[data["model_normalized"] == model1.strip().lower()]
    cars2 = data[data["model_normalized"] == model2.strip().lower()]
    if cars1.empty and cars2.empty:
        return f"Neither '{model1}' nor '{model2}' were found in the dataset."
    elif cars1.empty:
        return f"Model '{model1}' was not found in the dataset."
    elif cars2.empty:
        return f"Model '{model2}' was not found in the dataset."
    return pd.concat([cars1.describe(), cars2.describe()], axis=1, keys=[model1, model2])

# Streamlit Interface
st.title("Unified Car Price Prediction and Chatbot")

# Chatbot Section
st.subheader("Chat with the Bot")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

user_input = st.text_input("Type your message:", key="chat_input")
if st.button("Send"):
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        responses = send_to_rasa(user_input)
        for response in responses:
            st.session_state.messages.append({"role": "bot", "content": response})

# Recommendation Section
st.subheader("Set Preferences for Recommendations")
budget = st.slider("Select your budget range (£):", 5000, 50000, (10000, 20000), step=1000)
fuel_type = st.selectbox("Select fuel type:", ["Any", "Petrol", "Diesel"])
max_mileage = st.number_input("Enter maximum mileage:", min_value=0, value=50000, step=1000)
transmission = st.selectbox("Select transmission type:", ["Any", "Manual", "Automatic"])

st.session_state.user_preferences = {
    "budget": budget,
    "fuelType": fuel_type if fuel_type != "Any" else None,
    "mileage": max_mileage,
    "transmission": transmission if transmission != "Any" else None
}

if st.button("Show Recommendations"):
    filtered_cars = filter_cars(data, st.session_state.user_preferences)
    if not filtered_cars.empty:
        st.write("### Recommended Cars:")
        st.write(filtered_cars[["model", "year", "price", "fuelType", "mileage", "transmission", "engineSize"]])
    else:
        st.write("No cars match your preferences. Please adjust your filters.")

# Comparison Section
st.subheader("Compare Two Car Models")
model1 = st.text_input("Enter first car model:")
model2 = st.text_input("Enter second car model:")
if st.button("Compare Models"):
    comparison = compare_models(data, model1, model2)
    if isinstance(comparison, str):
        st.write(comparison)
    else:
        st.write(comparison)
