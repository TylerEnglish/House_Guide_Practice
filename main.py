# Main.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scripts.fetch_data import fetch_data
from scripts.preprocess import load_and_clean_data, save_processed_data
from scripts.data_engineering import process_and_save
from scripts.model import train_models
from scripts.logic import predict_price_from_features, explain_prediction  # ensure pipeline is imported if needed
from scripts.charts import *
import os
import time
import plotly.express as px
import plotly.graph_objects as go
import shap

# ---------- Custom CSS for Dashboard Look ----------
st.markdown("""
<style>
    /* Main container */
    .main { background-color: #f8f9fa; padding: 2rem; }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Titles */
    .dashboard-title {
        font-size: 2.5rem;
        color: #1a237e;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1a237e;
        padding-bottom: 1rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        margin-bottom: 2rem;
    }
    
    /* Charts */
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)



def pipeline():
    fetch_data()
    df = load_and_clean_data()
    save_processed_data(df)
    process_and_save()
    train_models() 

# Define the path 
DATA_FILE = os.path.join(os.path.dirname(__file__), "derived_data", "engineered_boston_housing.csv")

# Run the pipeline 
if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
    pipeline()
    

# ---------- Data Pipeline ----------
DATA_FILE = os.path.join(os.path.dirname(__file__), "derived_data", "engineered_boston_housing.csv")
if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
    with st.spinner("Running data pipeline..."):
        pipeline()

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

df = load_data()

# ---------- Dashboard Layout ----------
st.title("🏘️ Boston Real Estate Analytics Dashboard")

# Sidebar Controls (common to all modes)
with st.sidebar:
    st.header("🔍 Filters & Controls")
    selected_mode = st.radio("Dashboard Mode", ["Market Analysis", "Price Prediction"])
    
    with st.expander("📍 Location Filters", expanded=True):
        locations = st.multiselect("Select Areas", df['RAD'].unique(), default=df['RAD'].unique())
    
    with st.expander("🏠 Property Features for Filtering", expanded=True):
        room_range = st.slider("Number of Rooms", float(df['RM'].min()), float(df['RM'].max()), (4.0, 8.0))
        age_range = st.slider("Property Age", int(df['AGE'].min()), int(df['AGE'].max()), (20, 80))

# Filter data based on selections
filtered_df = df[
    (df['RAD'].isin(locations)) &
    (df['RM'].between(room_range[0], room_range[1])) &
    (df['AGE'].between(age_range[0], age_range[1]))
]

# ---------- Price Prediction Inputs in Sidebar (when selected) ----------
if selected_mode == "Price Prediction":
    st.sidebar.markdown("### 🏠 Price Prediction Inputs")
    # For each feature expected by the model, create an input widget.
    # Adjust ranges/defaults as necessary.
    user_crim    = st.sidebar.number_input("Crime Rate (CRIM)", min_value=0.0, max_value=100.0, value=float(df['CRIM'].median()))
    user_zn      = st.sidebar.number_input("Residential Land Zoned (ZN)", min_value=0.0, max_value=100.0, value=float(df['ZN'].median()))
    user_indus   = st.sidebar.number_input("Non-retail Business Acres (INDUS)", min_value=0.0, max_value=30.0, value=float(df['INDUS'].median()))
    user_chas    = st.sidebar.selectbox("Charles River Dummy (CHAS)", options=[0, 1], index=0)
    user_nox     = st.sidebar.number_input("Nitric Oxides Concentration (NOX)", min_value=0.0, max_value=1.0, value=float(df['NOX'].median()))
    user_rm      = st.sidebar.number_input("Average Number of Rooms (RM)", min_value=0.0, max_value=10.0, value=float(df['RM'].median()))
    user_age     = st.sidebar.number_input("Proportion of Owner-occupied Units Built Prior (AGE)", min_value=0, max_value=100, value=int(df['AGE'].median()))
    user_dis     = st.sidebar.number_input("Weighted Distances (DIS)", min_value=0.0, max_value=20.0, value=float(df['DIS'].median()))
    user_rad     = st.sidebar.number_input("Index of Accessibility to Highways (RAD)", min_value=1, max_value=10, value=int(df['RAD'].median()))
    user_tax     = st.sidebar.number_input("Full-value Property-tax Rate (TAX)", min_value=0, max_value=1000, value=int(df['TAX'].median()))
    user_ptratio = st.sidebar.number_input("Pupil-Teacher Ratio (PTRATIO)", min_value=0.0, max_value=30.0, value=float(df['PTRATIO'].median()))
    user_b       = st.sidebar.number_input("1000(Bk - 0.63)^2 (B)", min_value=0.0, max_value=500.0, value=float(df['B'].median()))
    user_lstat   = st.sidebar.number_input("Lower Status of the Population (LSTAT)", min_value=0.0, max_value=50.0, value=float(df['LSTAT'].median()))
    
    # Collect the inputs in a dictionary:
    user_features = {
        'CRIM': user_crim,
        'ZN': user_zn,
        'INDUS': user_indus,
        'CHAS': user_chas,
        'NOX': user_nox,
        'RM': user_rm,
        'AGE': user_age,
        'DIS': user_dis,
        'RAD': user_rad,
        'TAX': user_tax,
        'PTRATIO': user_ptratio,
        'B': user_b,
        'LSTAT': user_lstat
    }

# ---------- Market Analysis Layout ----------
st.markdown("""
<div class="dashboard-title">
    Market Overview
</div>
""", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown("""
    <div class="metric-card">
        <h3>📈 Avg Price</h3>
        <h2>${:,.0f}</h2>
    </div>
    """.format(filtered_df['price'].mean()), unsafe_allow_html=True)

with kpi2:
    st.markdown("""
    <div class="metric-card">
        <h3>🏘️ Avg Rooms</h3>
        <h2>{:.1f}</h2>
    </div>
    """.format(filtered_df['RM'].mean()), unsafe_allow_html=True)

with kpi3:
    st.markdown("""
    <div class="metric-card">
        <h3>👮♂️ Crime Rate</h3>
        <h2>{:.2f}</h2>
    </div>
    """.format(filtered_df['CRIM'].mean()), unsafe_allow_html=True)

with kpi4:
    st.markdown("""
    <div class="metric-card">
        <h3>🏛️ Tax Rate</h3>
        <h2>{:,.0f}</h2>
    </div>
    """.format(filtered_df['TAX'].mean()), unsafe_allow_html=True)

# ---------- Main Charts ----------
col1, col2 = st.columns([3, 2])
with col1:
    with st.container():
        st.markdown("### 📊 Price Distribution by Location")
        fig = px.box(filtered_df, x='RAD', y='price', color='RAD',
                     color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    with st.container():
        st.markdown("### 📍 Price vs Crime Rate")
        fig = px.scatter(filtered_df, x='CRIM', y='price', color='RAD',
                         trendline="lowess", 
                         color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(fig, use_container_width=True)

# ---------- Interactive Market Trends ----------
st.markdown("### 📅 Historical Price Trends")
trend_col1, trend_col2 = st.columns([2, 1])
with trend_col1:
    feature = st.selectbox("Select Trend Feature", ['RM', 'LSTAT', 'PTRATIO', 'AGE'])
    fig = px.scatter(filtered_df, x=feature, y='price', color='RAD',
                     trendline="lowess", 
                     color_discrete_sequence=px.colors.sequential.Blues_r)
    st.plotly_chart(fig, use_container_width=True)

with trend_col2:
    st.markdown("#### Correlation Matrix")
    corr_matrix = filtered_df[['price', 'RM', 'LSTAT', 'PTRATIO', 'AGE']].corr()
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

# ---------- Price Prediction Section (Main Area) ----------
if selected_mode == "Price Prediction":
    st.markdown("---")
    st.markdown("""
    <div class="dashboard-title">
        Price Prediction Engine
    </div>
    """, unsafe_allow_html=True)
    
    pred_col1, pred_col2 = st.columns([2, 1])
    with pred_col2:
        st.markdown("### 📈 Prediction Results")
        # Pass the complete features dictionary from the sidebar inputs:
        price = predict_price_from_features(user_features)
        st.markdown(f"""
        <div class="metric-card" style="background: #e3f2fd;">
            <h3>Predicted Price</h3>
            <h1>${price:,.0f}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # SHAP Explanation
        with st.expander("📊 Feature Impact Analysis"):
            try:
                explanation, base_value = explain_prediction(user_features, model_name="voting")
                import matplotlib.pyplot as plt
                plt.clf()  # Clear any existing figure
                # Limit the display to 13 features (the number of raw inputs)
                shap.plots.waterfall(explanation, max_display=13, show=False)
                st.pyplot(plt.gcf())
            except Exception as e:
                st.error(f"Error generating SHAP explanation: {e}")

# ---------- Similar Properties Section ----------
st.markdown("---")
st.markdown("""
<div class="dashboard-title">
    Comparable Properties
</div>
""", unsafe_allow_html=True)

similar_col1, similar_col2 = st.columns([2, 1])
with similar_col1:
    st.markdown("### 🗺️ Property Map")
    # Generate mock coordinates for demonstration
    mock_data = filtered_df.copy().sample(10)
    mock_data['lat'] = np.random.uniform(42.3, 42.4, size=10)
    mock_data['lon'] = np.random.uniform(-71.1, -71.0, size=10)
    st.map(mock_data[['lat', 'lon', 'price']].rename(columns={'price': 'size'}))

with similar_col2:
    st.markdown("### 🏆 Top Listings")
    top_listings = filtered_df.nlargest(5, 'price')[['RM', 'price', 'AGE']]
    st.dataframe(
        top_listings.style.format({'price': '${:,.0f}'}),
        height=300
    )

# ---------- Raw Data Section ----------
with st.expander("📁 View Raw Data"):
    st.dataframe(filtered_df.sort_values('price', ascending=False), height=300)