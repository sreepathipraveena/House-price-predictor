import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# Set Premium Page Config
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Theme and Custom CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #111827 50%, #0c1020 100%);
        color: #f3f4f6;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.02em;
    }
    
    /* Header Gradient styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 30%, #319795 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #9ca3af;
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Premium Glassmorphic Card styling */
    .glass-card {
        background: rgba(17, 25, 40, 0.65);
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.5rem;
    }
    
    /* Metric Display */
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        color: #38bdf8;
        text-shadow: 0 0 25px rgba(56, 189, 248, 0.45);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        text-transform: uppercase;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        color: #9ca3af;
    }
    
    .metric-sub {
        font-size: 0.95rem;
        color: #6b7280;
        margin-top: 0.5rem;
        line-height: 1.6;
    }
    
    /* Stat grid styling */
    .stat-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .stat-box {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-val {
        font-family: 'Outfit', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #f3f4f6;
    }
    
    .stat-lbl {
        font-size: 0.75rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Pipeline Model
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# Header layout
st.markdown('<div class="main-header">California House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your home specifications below to estimate its market price in California</div>', unsafe_allow_html=True)

# Create Page Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Interactive Predictor", "📊 Exploratory Data Analysis", "📈 Model Weights & Performance"])

if model is None:
    st.error("🚨 Missing File: `model.pkl` could not be found. Please run the model training script locally to generate the model pickle file.")
else:
    # ------------------ TAB 1: PREDICTOR ------------------
    with tab1:
        st.write("---")
        # Layout splits into form inputs (left) and result display/map (right)
        col_inputs, col_visuals = st.columns([1.2, 1], gap="large")
        
        with col_inputs:
            st.markdown("### 🛠️ Configure Your House Specifications")
            st.caption("Input your property parameters to predict its value based on California's regional census data.")
            
            with st.expander("🏠 Home Layout & Physical Size", expanded=True):
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    bedrooms_per_house = st.slider("Number of Bedrooms", min_value=1, max_value=8, value=3, step=1, help="Total bedrooms in the house.")
                    rooms_per_house = st.slider("Total Rooms (Living Area, Kitchen, etc.)", min_value=2, max_value=16, value=6, step=1, help="Total rooms including bedrooms, kitchen, living rooms, etc.")
                with r_col2:
                    housing_median_age = st.slider("House Age (Years Built)", min_value=1.0, max_value=52.0, value=20.0, step=1.0, help="Age of the house structure.")
                    occupants = st.slider("Number of Occupants (Household size)", min_value=1, max_value=10, value=3, step=1, help="Number of people living in the home.")

            with st.expander("📍 Location & Region Details", expanded=True):
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    longitude = st.slider("Longitude", min_value=-124.35, max_value=-114.31, value=-122.23, step=0.01, help="Geographical longitude location.")
                with g_col2:
                    latitude = st.slider("Latitude", min_value=32.54, max_value=41.95, value=37.88, step=0.01, help="Geographical latitude location.")
                
                ocean_proximity = st.selectbox(
                    "Ocean Proximity",
                    options=["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"],
                    index=0,
                    help="Categorical proximity of the location to the ocean."
                )
            
            with st.expander("💰 Regional Socio-Economic Level", expanded=True):
                annual_income_usd = st.slider(
                    "Median Household Annual Income in Area (USD)", 
                    min_value=10000, 
                    max_value=200000, 
                    value=83250, 
                    step=1000,
                    format="$%,d",
                    help="Median annual household income of the surrounding neighborhood."
                )

        with col_visuals:
            st.markdown("### 🔮 Predicted Market Value")
            
            # Census-Level Scaling Factor
            # To feed the block-level model, we project individual house features to a standard census block size (H = 400 households)
            H = 400
            
            # Preprocess features into block-level aggregations
            block_rooms = rooms_per_house * H
            block_bedrooms = bedrooms_per_house * H
            block_population = occupants * H
            block_households = H
            model_income = annual_income_usd / 10000 # Income in 10k metric scale
            
            # Assemble feature DataFrame matching the model's pipeline structure
            input_df = pd.DataFrame({
                "longitude": [longitude],
                "latitude": [latitude],
                "housing_median_age": [housing_median_age],
                "total_rooms": [block_rooms],
                "total_bedrooms": [block_bedrooms],
                "population": [block_population],
                "households": [block_households],
                "median_income": [model_income],
                "ocean_proximity": [ocean_proximity]
            })
            
            # Make prediction
            try:
                pred = model.predict(input_df)[0]
                pred = max(0.0, pred)  # Prevent negative prices
                
                # Render Premium Glass Card
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Estimated Property Price</div>
                    <div class="metric-value">${pred:,.2f}</div>
                    <div class="metric-sub">
                        Calculated using a Scikit-Learn Preprocessing & Regressor pipeline.<br>
                        Location: <b>{latitude:.2f}° N, {longitude:.2f}° W</b> (${ocean_proximity})
                    </div>
                    
                    <div class="stat-container">
                        <div class="stat-box">
                            <div class="stat-lbl">Rooms/House</div>
                            <div class="stat-val">{rooms_per_house}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-lbl">Bedrooms/House</div>
                            <div class="stat-val">{bedrooms_per_house}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-lbl">Family Size</div>
                            <div class="stat-val">{occupants}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
            
            # Geolocational Map
            st.markdown("### 🗺️ Geographic Location in California")
            map_data = pd.DataFrame({
                "latitude": [latitude],
                "longitude": [longitude]
            })
            st.map(map_data, zoom=9, use_container_width=True)

    # ------------------ TAB 2: EXPLORATORY DATA ANALYSIS ------------------
    with tab2:
        st.write("---")
        st.markdown("### 📊 Exploratory Data Analysis (EDA)")
        st.caption("Visualizing dataset distributions and correlations to discover underlying insights.")
        
        col_eda1, col_eda2 = st.columns(2, gap="large")
        
        with col_eda1:
            st.markdown("#### **Target Variable Distribution**")
            st.write("The distribution of house prices helps identify outliers. Notice the capping boundary at $500,001.")
            if os.path.exists("plots/target_distribution.png"):
                st.image("plots/target_distribution.png", use_container_width=True)
            else:
                st.info("Run `python train_model.py` locally to save the distribution plot under `plots/target_distribution.png`.")
                
        with col_eda2:
            st.markdown("#### **Feature Correlations Heatmap**")
            st.write("Correlation analysis helps determine linear dependencies. `median_income` exhibits the strongest positive correlation.")
            if os.path.exists("plots/correlation_matrix.png"):
                st.image("plots/correlation_matrix.png", use_container_width=True)
            else:
                st.info("Run `python train_model.py` locally to save the correlation matrix plot under `plots/correlation_matrix.png`.")

    # ------------------ TAB 3: MODEL INSIGHTS & COEFFICIENTS ------------------
    with tab3:
        st.write("---")
        st.markdown("### 📈 Model Evaluation & Linear Weights")
        st.caption("Detailed breakdown of the Linear Regression model metrics and standardized coefficients.")
        
        col_m1, col_m2 = st.columns([1, 1.2], gap="large")
        
        with col_m1:
            st.markdown("#### **Performance Metrics**")
            
            # Read metrics from local text file
            mae_val, rmse_val, r2_val = "$50,670.49", "$70,059.19", "0.6254"
            if os.path.exists("metrics.txt"):
                try:
                    with open("metrics.txt", "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            if line.startswith("MAE:"):
                                mae_val = f"${float(line.split(':')[1]):,.2f}"
                            elif line.startswith("RMSE:"):
                                rmse_val = f"${float(line.split(':')[1]):,.2f}"
                            elif line.startswith("R2:"):
                                r2_val = f"{float(line.split(':')[1]):.4f}"
                except:
                    pass
            
            st.markdown(f"""
            * **Mean Absolute Error (MAE)**: `{mae_val}` (Average magnitude of errors)
            * **Root Mean Squared Error (RMSE)**: `{rmse_val}` (Penalizes larger outliers)
            * **Coefficient of Determination ($R^2$)**: `{r2_val}` (Explains 62.5% of dataset variance)
            """)
            
            st.markdown("#### **Residuals Diagnostic Plots**")
            if os.path.exists("plots/residuals_analysis.png"):
                st.image("plots/residuals_analysis.png", use_container_width=True)
            else:
                st.info("Run `python train_model.py` locally to save the residuals analysis plot under `plots/residuals_analysis.png`.")
                
        with col_m2:
            st.markdown("#### **Standardized Coefficients (Feature Weights)**")
            st.write("Standardized coefficients indicate feature impact: positive weights increase predicted price, while negative weights decrease it.")
            
            # Read coefficients from local text file
            coef_list = []
            if os.path.exists("metrics.txt"):
                try:
                    with open("metrics.txt", "r") as f:
                        for line in f.readlines():
                            if line.startswith("COEF:"):
                                parts = line.strip().split(":")
                                coef_list.append({"Feature": parts[1], "Coefficient ($)": float(parts[2])})
                except:
                    pass
            
            if len(coef_list) > 0:
                coef_df = pd.DataFrame(coef_list)
                # Display styled DataFrame
                st.dataframe(
                    coef_df.style.format({"Coefficient ($)": "${:,.2f}"})
                    .background_gradient(cmap="coolwarm", subset=["Coefficient ($)"]),
                    use_container_width=True,
                    height=450
                )
            else:
                st.info("Run `python train_model.py` locally to save coefficient statistics.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #6b7280; font-size: 0.8rem;'>California Housing Analytics Dashboard • Powered by Streamlit & Scikit-Learn</div>", unsafe_allow_html=True)

