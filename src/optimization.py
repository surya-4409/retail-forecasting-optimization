import numpy as np
import pandas as pd
from scipy.stats import norm
import yaml
import os

class InventoryOptimizer:
    def __init__(self, config_path='../config.yaml'):
        # Load configuration
        if not os.path.exists(config_path):
            config_path = 'config.yaml'
            
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        inv_config = config.get('inventory', {})
        
        # Pull parameters dynamically from config
        self.lead_time = inv_config.get('lead_time_days', 3)
        self.holding_cost_rate = inv_config.get('holding_cost_annual_rate', 0.2) / 365 
        self.stockout_penalty = inv_config.get('stockout_penalty_per_unit', 15)

    def calculate_optimal_service_level(self):
        """Calculates the Critical Ratio (Newsvendor Model)"""
        co = self.holding_cost_rate * self.lead_time
        cu = self.stockout_penalty
        return cu / (cu + co)

    def generate_recommendations(self, forecast_df):
        """Calculates Safety Stock and Reorder Points."""
        target_service_level = self.calculate_optimal_service_level()
        z_score = norm.ppf(target_service_level)
        
        recs = []
        unique_pairs = forecast_df[['store_id', 'product_id']].drop_duplicates()
        
        for _, row in unique_pairs.iterrows():
            s_id = row['store_id']
            p_id = row['product_id']
            
            item_data = forecast_df[(forecast_df['store_id'] == s_id) & (forecast_df['product_id'] == p_id)]
            
            # Expected demand over lead time
            daily_forecast = item_data['ensemble_forecast'].mean()
            lead_time_demand = daily_forecast * self.lead_time
            
            # Standard deviation of demand
            std_demand = item_data['imputed_sales'].std()
            if pd.isna(std_demand):
                std_demand = 0
                
            # Safety stock = Z * std_dev * sqrt(lead_time)
            safety_stock = z_score * std_demand * np.sqrt(self.lead_time)
            
            # Reorder Point (ROP)
            rop = lead_time_demand + safety_stock
            
            recs.append({
                'store_id': s_id,
                'product_id': p_id,
                'avg_daily_forecast': round(daily_forecast, 2),
                'lead_time_demand': round(lead_time_demand, 2),
                'safety_stock': round(safety_stock, 0),
                'reorder_point': round(rop, 0),
                'recommended_order_qty': round(rop, 0)
            })
            
        return pd.DataFrame(recs)