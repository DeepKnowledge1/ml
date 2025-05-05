import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import plot_importance

# Set page config
st.set_page_config(page_title="🏠 House Price Predictor", layout="wide", page_icon="🏡")

# Load CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("./Supervised_Learning/Real_project/House_price/style.css")  # Optional: Add custom styles in style.css

# Title & Header
st.title("🏡 The Accuarate Melbourne House Price Prediction App")
st.markdown("Built with XGBoost and Streamlit — Predict house prices interactively!")

# Sidebar
st.sidebar.header("🔍 Input Features")
st.sidebar.markdown("Adjust the values below to predict house price.")

# Load dataset (for feature info)
@st.cache_data
def load_data():
    df = pd.read_csv("./Supervised_Learning/Real_project/House_price/MELBOURNE_HOUSE_PRICES_LESS.csv")
    return df

df = load_data()

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("./Supervised_Learning/Real_project/House_price/xgboost_house_price_model.pkl")
    return model

model = load_model()

# Sidebar Inputs
rooms = st.sidebar.slider("🛏 Rooms", min_value=1, max_value=10, value=3)
property_type = st.sidebar.selectbox("🏠 Property Type", options=["h - House", "u - Unit", "t - Townhouse"])
postcode = st.sidebar.number_input("📬 Postcode", min_value=3000, max_value=3999, value=3000)
distance = st.sidebar.slider("📏 Distance to City (km)", min_value=0.0, max_value=50.0, value=10.2, step=0.1)
prop_count = st.sidebar.number_input("🏘 Property Count in Suburb", min_value=100, max_value=10000, value=1000)

# Encode property type
type_map = {"h - House": 0, "u - Unit": 1, "t - Townhouse": 2}
type_encoded = type_map[property_type]

# Prepare input array
input_data = np.array([[rooms, type_encoded, postcode, distance, prop_count]])

# Make prediction
prediction = model.predict(input_data)[0]
formatted_prediction = "${:,.2f}".format(prediction)

# Main Panel
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📊 Your Input Features")
    input_df = pd.DataFrame({
        'Feature': ['Rooms', 'Property Type', 'Postcode', 'Distance to City (km)', 'Property Count'],
        'Value': [rooms, property_type.split(" - ")[1], postcode, distance, prop_count]
    })
    st.table(input_df)

with col2:
    st.subheader("🔮 Predicted House Price")
    st.markdown(f"<h2 style='color:#4CAF50;'>{formatted_prediction}</h2>", unsafe_allow_html=True)
    st.info("This is an estimate based on the selected features and our trained XGBoost model.")

# Additional Info
# st.markdown("---")

# fig, ax = plt.subplots()  # Create a Matplotlib figure
# sns.histplot(df['Price'], kde=True, ax=ax)  # Plot on the created axis
# ax.set_title("Distribution of House Prices")
# ax.set_xlabel("Price")
# ax.set_ylabel("Frequency")

# # Display in Streamlit
# st.pyplot(fig)  # Render the plot


# st.markdown("---")
# st.subheader("📈 Feature Importance")
# fig, ax = plt.subplots(figsize=(8, 4))
# plot_importance(model, ax=ax, max_num_features=5, color="#4285F4")
# st.pyplot(fig)

st.markdown("---")

# --- Layout: Two Columns ---
col1, col2 = st.columns(2)  # Creates 2 side-by-side columns


with col1:
    st.markdown("### Distribution of House Prices")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.histplot(df["Price"], kde=True, ax=ax1)
    ax1.set_xlabel("Price")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

# --- Plot 2: Feature Importance (Right Side) ---
with col2:
    st.markdown("### 📈 Feature Importance")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    plot_importance(model, ax=ax2, max_num_features=5, color="#4285F4")
    st.pyplot(fig2)


st.markdown("---")  # Horizontal line after plots
st.subheader("🧾 Dataset Sample")
st.write(df.head())

st.markdown("---")
st.markdown("© 2025 House Price Predictor | Built By Deep Knowledge")