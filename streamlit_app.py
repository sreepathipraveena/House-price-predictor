import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# Set Page Config
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #f7fafc;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    .predict-box {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .predict-val {
        font-size: 3rem;
        font-weight: 800;
        color: #38bdf8;
        text-shadow: 0px 0px 15px rgba(56, 189, 248, 0.4);
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# Title Section
st.title("🏠 California House Price Predictor")
st.markdown("Estimate the median house value of a block using a baseline **Linear Regression Pipeline** trained on the California Housing dataset.")
st.markdown("---")

if model is None:
    st.error("Error: `model.pkl` not found. Please make sure the model is trained and uploaded to the directory.")
else:
    # Set up columns for input and output
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.subheader("📍 Location & Geographic Features")
        
        # Latitude / Longitude Selectors
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            longitude = st.slider("Longitude (East/West)", min_value=-124.35, max_value=-114.31, value=-122.23, step=0.01)
        with subcol2:
            latitude = st.slider("Latitude (North/South)", min_value=32.54, max_value=41.95, value=37.88, step=0.01)
            
        ocean_proximity = st.selectbox(
            "Ocean Proximity Class",
            options=["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"],
            index=0
        )
        
        st.subheader("📊 Block Demographics & Physical attributes")
        
        subcol3, subcol4 = st.columns(2)
        with subcol3:
            housing_median_age = st.slider("Median House Age (Years)", min_value=1.0, max_value=52.0, value=41.0, step=1.0)
            total_rooms = st.number_input("Total Rooms in Block", min_value=10, max_value=50000, value=880)
            total_bedrooms = st.number_input("Total Bedrooms in Block", min_value=1, max_value=20000, value=129)
        with subcol4:
            annual_income = st.slider("Median Annual Income (USD)", min_value=5000, max_value=200000, value=83250, step=250)
            population = st.number_input("Block Population", min_value=5, max_value=50000, value=322)
            households = st.number_input("Total Households (Families)", min_value=5, max_value=20000, value=126)
            
    with col2:
        st.subheader("🔮 Price Prediction")
        
        # Format income for model input (tens of thousands USD)
        model_income = annual_income / 10000
        
        # Assemble feature DataFrame
        input_data = pd.DataFrame({
            "longitude": [longitude],
            "latitude": [latitude],
            "housing_median_age": [housing_median_age],
            "total_rooms": [total_rooms],
            "total_bedrooms": [total_bedrooms],
            "population": [population],
            "households": [households],
            "median_income": [model_income],
            "ocean_proximity": [ocean_proximity]
        })
        
        # Predict price
        try:
            pred = model.predict(input_data)[0]
            # Ensure price isn't negative
            pred = max(0.0, pred)
            
            # Predict Card
            st.markdown(f"""
            <div class="predict-box">
                <p style="text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.1em; color: #94a3b8;">Predicted Median House Value</p>
                <div class="predict-val">${pred:,.2f}</div>
                <p style="color: #94a3b8; font-size: 0.9rem;">
                    Location: <b>{latitude:.2f}° N, {longitude:.2f}° W</b> ({ocean_proximity})<br>
                    Income Level: <b>${annual_income:,.0f}/year</b> (${model_income:.4f} metric units)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error calculating prediction: {e}")
            
        # Map Display
        st.subheader("🗺️ Geographical Location Map")
        map_df = pd.DataFrame({
            "latitude": [latitude],
            "longitude": [longitude]
        })
        st.map(map_df, zoom=10, use_container_width=True)
        
st.markdown("---")
st.markdown("California Housing Model • Generated using Scikit-Learn Pipeline & Streamlit")

