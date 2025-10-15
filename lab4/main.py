import joblib
import pandas as pd
import streamlit as st
from keras.models import load_model


# Load previously dumped model, scaler, and column definitions
@st.cache_resource
def load_resources():
    model = load_model("lab1/diamonds.keras")
    scaler = joblib.load("lab1/scaler.pkl")
    columns = joblib.load("lab1/columns.pkl")
    return model, scaler, columns


model, scaler, columns = load_resources()
categorical_features = ['cut', 'color', 'clarity']
numeric_features = ['carat', 'x', 'y', 'z']


def get_column_categories(name: str):
    return [value.split("_")[1] for value in columns.values if value.startswith(name + "_")]


cut_choices = get_column_categories("cut")
color_choices = get_column_categories("color")
clarity_choices = get_column_categories("clarity")

st.title("Diamond Price Prediction")
st.header("The 4 Cs of Diamonds💎")
carat = st.slider("Carat (physical weight)", min_value=0.2, max_value=1.22, value=0.58, step=0.01)
cut = st.select_slider("Cut (gem fashioning)", options=cut_choices)
color = st.select_slider("Color (the whiter the better)", options=color_choices)
clarity = st.select_slider("Clarity (depends on number inclusions)", options=clarity_choices)
st.header("Dimensions📐")
x = st.slider("X", min_value=3.73, max_value=6.89, value=5.25, step=0.01)
y = st.slider("Y", min_value=3.68, max_value=6.77, value=5.25, step=0.01)
z = st.slider("Z", min_value=1.07, max_value=3.62, value=3.23, step=0.01)


def preprocess_input():
    df = pd.DataFrame({
        'carat': [carat],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity],
        'x': [x],
        'y': [y],
        'z': [z],
    })
    # One-hot encoding of categorical variables
    df = pd.get_dummies(df, columns=categorical_features)
    # Add missing columns that exist in X_train but not in new_data
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    # Reorder columns to match training data
    df = df[columns]
    # Normalize numeric features (using the same scaler from training)
    df[numeric_features] = scaler.transform(df[numeric_features])
    return df


# Run whenever some feature is changed
if any([carat, cut, color, clarity, x, y, z]):
    inputs = preprocess_input()
    predictions = model.predict(inputs)
    st.success(f"Estimated gem price: **${predictions[0][0]:.2f}**")
