import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set page config for a wider, modern layout
st.set_page_config(page_title="Retail Forecasting Dashboard", layout="wide")

# Main Title and Description
st.title("📦 Retail Demand Forecasting & Inventory Optimization")
st.markdown("This dashboard provides product-store level demand forecasts and automated inventory recommendations based on our ensemble machine learning pipeline.")

# Load the generated recommendations data
@st.cache_data
def load_data():
    # Smart pathing depending on where the user runs the script from
    base_path = "data" if os.path.exists("data") else "../data"
    file_path = f"{base_path}/final_inventory_recommendations.csv"
    
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

if df is None:
    st.error("Data file not found. Please ensure you have run the modeling and optimization pipeline.")
else:
    # --- Sidebar Configuration ---
    st.sidebar.header("🔍 Filters")
    store_selection = st.sidebar.selectbox("Select Store", sorted(df['store_id'].unique()))
    
    filtered_df_store = df[df['store_id'] == store_selection]
    product_selection = st.sidebar.selectbox("Select Product", sorted(filtered_df_store['product_id'].unique()))
    
    # Apply filters to get the specific time series
    final_df = filtered_df_store[filtered_df_store['product_id'] == product_selection].sort_values('date')
    
    # Developer Identity cleanly placed at the bottom of the sidebar
    st.sidebar.divider()
    st.sidebar.caption("👨‍💻 **Developed by:**")
    st.sidebar.caption("Surya (Roll No: 23MH1A4409)")
    st.sidebar.caption("**Role:** Data Scientist")

    # --- Main Workspace ---
    
    # Top Level Metrics
    st.header(f"Inventory Recommendations: {store_selection} - {product_selection}")
    
    # Get the latest recommendation (the first day of the forecast period)
    latest_rec = final_df.iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recommended Order Qty", f"{int(latest_rec['recommended_order_quantity'])} units")
    col2.metric("Safety Stock", f"{int(latest_rec['safety_stock'])} units")
    col3.metric("Reorder Point", f"{int(latest_rec['reorder_point'])} units")
    col4.metric("Avg Daily Forecast", f"{round(final_df['ensemble_forecast'].mean(), 1)} units")
    
    st.divider()
    
    # Interactive Chart
    st.subheader("Forecasted Demand vs Dynamic Reorder Point")
    
    # Create an interactive Plotly chart
    fig = px.line(final_df, x='date', y='ensemble_forecast', 
                  labels={'ensemble_forecast': 'Forecasted Daily Demand', 'date': 'Date'},
                  title="60-Day Forward Look")
    
    # Add the Reorder Point as a secondary red dashed line
    fig.add_scatter(x=final_df['date'], y=final_df['reorder_point'], 
                    mode='lines', name='Reorder Point (ROP)', 
                    line=dict(dash='dash', color='red'))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Data Table
    st.subheader("Detailed Daily Projections")
    
    # Show a clean table for the business user to export if needed
    display_cols = ['date', 'ensemble_forecast', 'safety_stock', 'reorder_point', 'recommended_order_quantity']
    clean_df = final_df[display_cols].copy()
    
    # Round numbers for clean display
    for col in display_cols[1:]:
        clean_df[col] = clean_df[col].round(1)
        
    st.dataframe(clean_df.set_index('date'), use_container_width=True)