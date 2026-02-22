import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config for a wider, modern layout
st.set_page_config(page_title="Retail Forecasting Dashboard", layout="wide")

st.title("📦 Retail Demand Forecasting & Inventory Optimization")
st.markdown("This dashboard provides product-store level demand forecasts, automated inventory recommendations, and historical performance tracking based on our ensemble machine learning pipeline.")

# Load both future projections and historical track record
@st.cache_data
def load_data():
    base_path = "data" if os.path.exists("data") else "../data"
    future_path = f"{base_path}/final_inventory_recommendations.csv"
    history_path = f"{base_path}/historical_performance.csv"
    
    df_future = pd.read_csv(future_path) if os.path.exists(future_path) else None
    df_history = pd.read_csv(history_path) if os.path.exists(history_path) else None
    
    if df_future is not None: df_future['date'] = pd.to_datetime(df_future['date'])
    if df_history is not None: df_history['date'] = pd.to_datetime(df_history['date'])
        
    return df_future, df_history

df_future, df_history = load_data()

if df_future is None or df_history is None:
    st.error("Data files not found. Please ensure you have run the modeling and optimization pipeline.")
else:
    # --- Sidebar Configuration ---
    st.sidebar.header("🔍 Filters")
    store_selection = st.sidebar.selectbox("Select Store", sorted(df_future['store_id'].unique()))
    
    filtered_future = df_future[df_future['store_id'] == store_selection]
    product_selection = st.sidebar.selectbox("Select Product", sorted(filtered_future['product_id'].unique()))
    
    # Apply filters to get specific time series
    final_future = filtered_future[filtered_future['product_id'] == product_selection].sort_values('date')
    final_history = df_history[(df_history['store_id'] == store_selection) & (df_history['product_id'] == product_selection)].sort_values('date')
    
    # Developer Identity
    st.sidebar.divider()
    st.sidebar.caption("👨‍💻 **Developed by:**")
    st.sidebar.caption("Surya (Roll No: 23MH1A4409)")
    st.sidebar.caption("**Role:** Data Scientist")

    # --- Main Workspace ---
    st.header(f"Inventory Intelligence: {store_selection} - {product_selection}")
    
    # Create Tabs to satisfy the "Historical Performance" requirement
    tab1, tab2 = st.tabs(["🔮 Future Projections & Orders", "📊 Historical Model Accuracy"])
    
    with tab1:
        # Top Level Metrics
        latest_rec = final_future.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Recommended Order Qty", f"{int(latest_rec['recommended_order_quantity'])} units")
        col2.metric("Safety Stock", f"{int(latest_rec['safety_stock'])} units")
        col3.metric("Reorder Point", f"{int(latest_rec['reorder_point'])} units")
        col4.metric("Avg Daily Forecast", f"{round(final_future['ensemble_forecast'].mean(), 1)} units")
        
        st.divider()
        st.subheader("Forecasted Demand with Uncertainty Intervals")
        
        # Advanced Plotly Chart with Prediction Intervals
        fig = go.Figure()
        
        # Add Upper Bound
        fig.add_trace(go.Scatter(x=final_future['date'], y=final_future['upper_bound'], 
                                 mode='lines', line=dict(width=0), showlegend=False))
        # Add Lower Bound with fill
        fig.add_trace(go.Scatter(x=final_future['date'], y=final_future['lower_bound'], 
                                 mode='lines', line=dict(width=0), fill='tonexty', 
                                 fillcolor='rgba(68, 122, 219, 0.2)', name='95% Prediction Interval'))
        
        # Add Core Forecast
        fig.add_trace(go.Scatter(x=final_future['date'], y=final_future['ensemble_forecast'], 
                                 mode='lines', name='Forecasted Demand', line=dict(color='#5D8AA8')))
        
        # Add Reorder Point
        fig.add_trace(go.Scatter(x=final_future['date'], y=final_future['reorder_point'], 
                                 mode='lines', name='Reorder Point (ROP)', line=dict(dash='dash', color='#E63946')))
        
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Table
        st.subheader("Detailed Daily Projections")
        display_cols = ['date', 'ensemble_forecast', 'safety_stock', 'reorder_point', 'recommended_order_quantity']
        clean_df = final_future[display_cols].copy()
        for col in display_cols[1:]: clean_df[col] = clean_df[col].round(1)
        st.dataframe(clean_df.set_index('date'), use_container_width=True)
        
    with tab2:
        st.subheader("Historical Tracking (Walk-Forward CV Results)")
        
        # Dynamically calculate the historical MAPE for the selected product
        actuals = final_history['imputed_sales']
        preds = final_history['ensemble_forecast']
        mape = (abs(actuals - preds) / (actuals + 1e-5)).mean() * 100
        
        # Dynamic Visual Indicator
        if mape <= 20:
            st.success(f"✅ **Model performing well on this item.** Historical MAPE: **{mape:.1f}%** (Target: < 20%)")
        else:
            st.warning(f"⚠️ **High error variance detected.** Historical MAPE: **{mape:.1f}%** (This is expected for highly intermittent items)")
            
        # Plot historical actuals vs predicted
        fig_hist = px.line(final_history, x='date', y=['imputed_sales', 'ensemble_forecast'],
                           labels={'value': 'Units', 'date': 'Date', 'variable': 'Legend'},
                           title="Historical Actuals vs. Model Predictions",
                           color_discrete_map={'imputed_sales': '#E63946', 'ensemble_forecast': '#5D8AA8'})
        
        # Clean up legend
        newnames = {'imputed_sales': 'Actual Sales', 'ensemble_forecast': 'Predicted Sales'}
        fig_hist.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
        
        fig_hist.update_layout(template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)