# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# --- Load artifacts (Cached!) ---
@st.cache_resource
def load_artifacts():
    """Loads the model and transformers once, keeping them in memory."""
    model = tf.keras.models.load_model("artifacts/housing_model.h5", compile=False)
    scaler = joblib.load("artifacts/scaler.pkl")
    
    # FIX: Ignore the broken feature_names.pkl file entirely. 
    # Extract the exact one-hot encoded columns the scaler memorized!
    features = scaler.feature_names_in_
    
    return model, scaler, features

# Unpack the cached artifacts
model, scaler, features = load_artifacts()

# --- Streamlit UI ---
st.title("🏠 Hamilton Housing Appraiser")

st.write(
    "Enter property details to predict the appraised value in Hamilton."
)

# Input widgets
calc_acres = st.number_input("Lot Size (Acres)", min_value=0.0, step=0.01, value=0.5)

# Get unique options from saved feature names
land_use_options = [f.replace("LAND_USE_CODE_DESC_", "") 
                    for f in features if f.startswith("LAND_USE_CODE_DESC_")]
land_use_options = ["Unknown"] + land_use_options

property_type_options = [f.replace("PROPERTY_TYPE_CODE_DESC_", "") 
                         for f in features if f.startswith("PROPERTY_TYPE_CODE_DESC_")]
property_type_options = ["Unknown"] + property_type_options

land_use = st.selectbox("Land Use", land_use_options)
property_type = st.selectbox("Property Type", property_type_options)

# --- Prepare input for prediction ---
def preprocess_input(calc_acres, land_use, property_type):
    # Initialize dictionary with zeros
    input_dict = {feat: 0 for feat in features}
    input_dict["CALC_ACRES"] = calc_acres
    
    # Set categorical flags
    land_feat = f"LAND_USE_CODE_DESC_{land_use}"
    prop_feat = f"PROPERTY_TYPE_CODE_DESC_{property_type}"
    
    if land_feat in input_dict:
        input_dict[land_feat] = 1
    if prop_feat in input_dict:
        input_dict[prop_feat] = 1
    
    # Convert to dataframe and explicitly enforce the column order 
    # to perfectly match the training features
    df_input = pd.DataFrame([input_dict], columns=features)
    
    # Scale numeric inputs
    df_input_scaled = scaler.transform(df_input)
    return df_input_scaled

# --- Prediction ---
if st.button("Predict Appraised Value"):
    # Add a visual spinner while the model predicts
    with st.spinner("Calculating appraisal..."):
        X_input = preprocess_input(calc_acres, land_use, property_type)
        # The [0][0] extracts the single scalar value from the 2D prediction array
        prediction = model.predict(X_input)[0][0]
    
    st.success(f"Estimated Appraised Value: ${prediction:,.2f}")


