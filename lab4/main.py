import os
import joblib
import pandas as pd
import streamlit as st
from keras.models import load_model


# === Load model, scaler, and column definitions ===
@st.cache_resource
def load_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "diamonds.keras")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    columns_path = os.path.join(base_dir, "columns.pkl")

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    columns = joblib.load(columns_path)

    return model, scaler, columns


model, scaler, feature_columns = load_resources()

# === Define feature types ===
categorical_features = ["cut", "color", "clarity"]
numerical_features = ["carat", "x", "y", "z"]


def get_category_options(prefix: str):
    """Return possible category values for a given prefix (e.g. 'cut')."""
    return [col.split("_")[1] for col in feature_columns if col.startswith(prefix + "_")]


# === UI section ===
st.title("💎 Diamond Price Prediction")
st.header("Diamond Parameters")

carat_value = st.slider("Carat weight", min_value=0.2, max_value=1.51, value=0.55, step=0.01)
cut_value = st.select_slider("Cut quality", options=get_category_options("cut"))
color_value = st.select_slider("Color grade", options=get_category_options("color"))
clarity_value = st.select_slider("Clarity", options=get_category_options("clarity"))

st.header("Dimensions (mm)")
x_value = st.slider("Length (X)", min_value=3.0, max_value=7.0, value=5.0, step=0.01)
y_value = st.slider("Width (Y)", min_value=3.0, max_value=7.0, value=5.0, step=0.01)
z_value = st.slider("Depth (Z)", min_value=1.0, max_value=4.0, value=3.0, step=0.01)


def preprocess_input():
    """Prepare input data for model prediction."""
    input_df = pd.DataFrame({
        "carat": [carat_value],
        "cut": [cut_value],
        "color": [color_value],
        "clarity": [clarity_value],
        "x": [x_value],
        "y": [y_value],
        "z": [z_value],
    })

    # One-hot encode categorical features
    input_df = pd.get_dummies(input_df, columns=categorical_features)

    # Add any missing columns (present in training data but absent in input)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training data
    input_df = input_df[feature_columns]

    # Normalize numerical features
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    return input_df


# === Prediction section ===
if any([carat_value, cut_value, color_value, clarity_value, x_value, y_value, z_value]):
    input_data = preprocess_input()
    predicted_price = model.predict(input_data)
    st.success(f"💰 Estimated Diamond Price: **${predicted_price[0][0]:.2f}**")
