import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

st.set_page_config(page_title="Walmart Sales Forecast", layout="wide", page_icon="📊")

st.title("📊 Walmart Executive Sales Forecasting Dashboard")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv('walmart_cleaned_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def load_model():
    import pickle
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    df_clean = load_data()
    model_optimized = load_model()
except Exception as e:
    st.error(f"Error loading files. Ensure they are uploaded to your GitHub repository. Error: {e}")
    st.stop()

# Business Summary
st.subheader("💡 Core Business Discoveries")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("**The Holiday Surge**\\n\\nHoliday weeks generate an average of **$1,121,632**, compared to **$1,033,918** during regular weeks. That's a consistent **+$87,714 revenue boost** per store.")
with col2:
    st.success("**The Historical Anchor**\\n\\nOur machine learning model discovered that **`last_year` sales** is your #1 strongest predictor. Shopping habits follow strict annual routines.")
with col3:
    st.warning("**External Immunity**\\n\\nData shows shifts in **Temperature** or **Fuel Prices** have almost **zero impact** on final weekly revenue. Walmart is an insulated necessity.")

st.markdown("---")

# Interactive Sidebar Filter
st.sidebar.header("🛠️ Store Analytics Filter")
selected_store = st.sidebar.selectbox("Select a Store to Inspect:", sorted(df_clean['Store'].unique()))
store_df = df_clean[df_clean['Store'] == selected_store]
avg_weekly_sales = store_df['Weekly_Sales'].mean()
st.sidebar.metric(label=f"Avg Weekly Sales (Store {selected_store})", value=f"${avg_weekly_sales:,.2f}")

# Visual Charts
st.subheader("📈 Trend Visualizations")
tab1, tab2 = st.tabs(["📅 Seasonality & Holidays", "⚙️ Model Decision Drivers"])

with tab1:
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.write("#### Regular Weeks vs. Holiday Weeks Average")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Holiday_Flag', y='Weekly_Sales', data=df_clean, palette='Blues_d', ax=ax, hue='Holiday_Flag', legend=False)
        ax.set_xticklabels(['Regular Week', 'Holiday Week'])
        ax.set_ylabel('Average Sales ($)')
        st.pyplot(fig)
        
    with col_chart2:
        st.write("#### Year-Over-Year Monthly Performance")
        monthly_sales = df_clean.groupby([df_clean['Date'].dt.year, df_clean['Date'].dt.month])['Weekly_Sales'].mean().reset_index()
        monthly_sales.columns = ['Year', 'Month', 'Weekly_Sales']
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='Month', y='Weekly_Sales', hue='Year', data=monthly_sales, marker='o', palette='tab10', ax=ax)
        ax.set_xlabel('Month (1 = Jan, 12 = Dec)')
        ax.set_ylabel('Sales ($)')
        ax.grid(True, linestyle='--')
        st.pyplot(fig)

with tab2:
    st.write("#### What Variables Direct the Artificial Intelligence Forecast?")
    features_list = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'last_week', 'two_weeks_ago', 'last_month', 'quater_month', 'last_year', 'Year', 'Month', 'Week']
    importance_df = pd.DataFrame({
        'Feature': features_list,
        'Importance': model_optimized.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='dodgerblue')
    ax.set_xlabel('Importance Score')
    st.pyplot(fig)

st.markdown("---")

# Predictive Calculator Form
st.subheader("🔮 Run a Live Sales Forecast")
pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    inp_holiday = st.selectbox("Is it a Holiday Week?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    inp_temp = st.number_input("Expected Temperature (°F):", value=60.0)
    inp_fuel = st.number_input("Expected Fuel Price ($/Gal):", value=3.50)

with pred_col2:
    inp_cpi = st.number_input("Current CPI (Inflation Index):", value=220.0)
    inp_unemp = st.number_input("Local Unemployment Rate (%):", value=7.0)
    inp_week = st.slider("Week Number of the Year:", min_value=1, max_value=52, value=45)

with pred_col3:
    inp_last_week = st.number_input("Sales Last Week ($):", value=float(store_df['last_week'].median()))
    inp_last_year = st.number_input("Sales Exactly 1 Year Ago ($):", value=float(store_df['last_year'].median()))

inp_two_weeks = store_df['two_weeks_ago'].median()
inp_last_month = store_df['last_month'].median()
inp_quarter = store_df['quater_month'].median()
inp_month = int(np.clip(inp_week // 4.3, 1, 12))
inp_year = 2012

if st.button("🚀 Calculate Future Revenue Estimate"):
    input_features = np.array([[
        selected_store, inp_holiday, inp_temp, inp_fuel, inp_cpi, inp_unemp,
        inp_last_week, inp_two_weeks, inp_last_month, inp_quarter, inp_last_year,
        inp_year, inp_month, inp_week
    ]])
    prediction = model_optimized.predict(input_features)[0]
    st.balloons()
    st.success(f"### Predicted Weekly Revenue: **${prediction:,.2f}**")
