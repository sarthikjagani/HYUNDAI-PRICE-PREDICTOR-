import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Price Prediction", page_icon="💰", layout="wide")

# Load data function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("hyundi.csv")
        return df
    except FileNotFoundError:
        st.error("Error: hyundi.csv file not found!")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess data for machine learning."""
    num_features = ["year", "mileage", "tax(£)", "mpg", "engineSize"]
    cat_features = ["model", "transmission", "fuelType"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_features),
        ],
        verbose_feature_names_out=False
    )
    
    features = num_features + cat_features
    target = "price"
    return preprocessor, features, target, num_features, cat_features

def get_feature_importance(model, X, preprocessor, num_features, cat_features):
    """Extract feature importance with correct feature names."""
    # Get feature names after preprocessing
    feature_names = (
        num_features +
        [f"{feat}_{val}" for feat, vals in zip(
            cat_features,
            preprocessor.named_transformers_['cat'].categories_
        ) for val in vals]
    )
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame with correct feature names
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    return feature_importance.sort_values('importance', ascending=False)

def train_model(df, preprocessor, features, target):
    """Train model using Random Forest with cross-validation."""
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use Random Forest Regressor (you can change to GradientBoostingRegressor if needed)
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "R²": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "CV_scores": cv_scores,
        "CV_mean": cv_scores.mean(),
        "CV_std": cv_scores.std()
    }
    
    return model, metrics, X_test, y_test, y_pred, X_train

def plot_residuals(y_test, y_pred):
    """Plot residuals analysis."""
    residuals = y_test - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted Values')
    
    # Residuals distribution
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_title('Residuals Distribution')
    
    return fig

def plot_price_distribution(df):
    """Create price distribution plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Regular histogram
    sns.histplot(data=df, x="price", bins=50, ax=ax1)
    ax1.set_title("Price Distribution")
    ax1.set_xlabel("Price (£)")
    ax1.set_ylabel("Count")
    
    # Log-transformed histogram for better visualization of skewed data
    sns.histplot(data=df, x="price", bins=50, ax=ax2, log_scale=True)
    ax2.set_title("Price Distribution (Log Scale)")
    ax2.set_xlabel("Price (£)")
    ax2.set_ylabel("Count")
    
    plt.tight_layout()
    return fig

def main():
    st.title("Hyundai Car Price Prediction 💰")
    
    df = load_data()
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["Price Prediction", "Model Performance", "Feature Analysis"])
        
        with tab1:
            st.header("Car Price Prediction")
            
            # Input columns for user-provided features
            col1, col2 = st.columns(2)
            
            # If the dataset contains only one unique model, display it instead of a select box.
            unique_models = df["model"].unique()
            with col1:
                if len(unique_models) == 1:
                    st.write(f"**Model:** {unique_models[0]}")
                    model_name = unique_models[0]
                else:
                    model_name = st.selectbox("Model", options=unique_models)
                year = st.slider("Year", min_value=2000, max_value=2025, value=2020)
                mileage = st.slider("Mileage (km)", min_value=0, max_value=300000, value=50000)
                transmission = st.selectbox("Transmission", options=df["transmission"].unique())
            
            with col2:
                fuel_type = st.selectbox("Fuel Type", options=df["fuelType"].unique())
                tax = st.slider("Tax (£)", min_value=0, max_value=500, value=100)
                mpg = st.slider("MPG", min_value=0, max_value=100, value=50)
                engine_size = st.slider("Engine Size (L)", min_value=0.0, max_value=5.0, value=1.5)

            # Prepare data and model
            preprocessor, features, target, num_features, cat_features = preprocess_data(df)
            model, metrics, X_test, y_test, y_pred, X_train = train_model(
                df, preprocessor, features, target
            )

            # Make prediction
            if st.button("Predict Price", type="primary"):
                input_data = pd.DataFrame({
                    "model": [model_name],
                    "year": [year],
                    "mileage": [mileage],
                    "transmission": [transmission],
                    "fuelType": [fuel_type],
                    "tax(£)": [tax],
                    "mpg": [mpg],
                    "engineSize": [engine_size],
                })
                
                predicted_price = model.predict(input_data)[0]
                st.success(f"### Predicted Price: £{predicted_price:,.2f}")
        
        with tab2:
            st.header("Model Performance Metrics")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{metrics['R²']:.3f}")
            with col2:
                st.metric("Mean Absolute Error", f"£{metrics['MAE']:,.2f}")
            with col3:
                st.metric("Root Mean Squared Error", f"£{metrics['RMSE']:,.2f}")
            
            # Cross-validation results
            st.subheader("Cross-validation Results")
            st.write(f"Mean R² (CV): {metrics['CV_mean']:.3f} (±{metrics['CV_std']:.3f})")
            
            # Residuals plot
            st.subheader("Residuals Analysis")
            residuals_fig = plot_residuals(y_test, y_pred)
            st.pyplot(residuals_fig)
            
            # Actual vs Predicted plot
            st.subheader("Actual vs Predicted Prices")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Actual Price")
            ax.set_ylabel("Predicted Price")
            ax.set_title("Actual vs Predicted Prices")
            st.pyplot(fig)
        
        with tab3:
            st.header("Feature Analysis")
            
            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = get_feature_importance(
                model['regressor'],
                X_train,
                preprocessor,
                num_features,
                cat_features
            )
            
            # Plot top 15 features
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
            plt.title("Top 15 Most Important Features")
            plt.xlabel("Importance")
            plt.tight_layout()
            st.pyplot(fig)
            
            # Correlation matrix for numerical features
            st.subheader("Feature Correlations")
            numerical_features = ["year", "mileage", "tax(£)", "mpg", "engineSize", "price"]
            correlation_matrix = df[numerical_features].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title("Correlation Matrix of Numerical Features")
            st.pyplot(fig)
            
            # Price Distribution Analysis
            st.subheader("Price Distribution Analysis")
            
            # Display basic statistics
            price_stats = df['price'].describe()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Price", f"£{price_stats['mean']:,.2f}")
            with col2:
                st.metric("Median Price", f"£{price_stats['50%']:,.2f}")
            with col3:
                st.metric("Min Price", f"£{price_stats['min']:,.2f}")
            with col4:
                st.metric("Max Price", f"£{price_stats['max']:,.2f}")
            
            # Price distribution plots
            price_dist_fig = plot_price_distribution(df)
            st.pyplot(price_dist_fig)
            
            # Additional price statistics by categories
            st.write("##### Price Distribution by Categories")
            col1, col2 = st.columns(2)
            
            with col1:
                # Average price by model
                model_prices = df.groupby('model')['price'].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                model_prices.plot(kind='bar')
                plt.title("Average Price by Model")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Average price by fuel type
                fuel_prices = df.groupby('fuelType')['price'].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                fuel_prices.plot(kind='bar')
                plt.title("Average Price by Fuel Type")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

if __name__ == "__main__":
    main()
