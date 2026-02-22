import numpy as np
import pandas as pd
from scipy.stats import norm

class InventoryOptimizer:
    def __init__(self, lead_time_days=3, holding_cost_annual_rate=0.2, stockout_penalty_per_unit=15):
        self.lead_time = lead_time_days
        # Convert annual holding cost rate to a daily rate
        self.holding_cost_rate = holding_cost_annual_rate / 365 
        self.stockout_penalty = stockout_penalty_per_unit

    def calculate_optimal_service_level(self, unit_cost):
        """Calculates target service level using the Newsvendor critical ratio."""
        # Overage cost: Cost to hold the item over the lead time
        holding_cost = unit_cost * self.holding_cost_rate * self.lead_time
        
        # Critical Ratio formula
        critical_ratio = self.stockout_penalty / (self.stockout_penalty + holding_cost)
        
        # Cap at 0.99 to avoid mathematically infinite safety stock
        return min(critical_ratio, 0.99)

    def generate_recommendations(self, forecast_df, residual_std):
        """Generates inventory recommendations based on forecasts and costs."""
        df = forecast_df.copy()
        
        # 1. Calculate dynamic service level per product based on its price
        df['target_service_level'] = df['base_price'].apply(self.calculate_optimal_service_level)
        df['z_score'] = norm.ppf(df['target_service_level'])
        
        # 2. Calculate Safety Stock (Uncertainty * Z-Score * sqrt(Lead Time))
        lead_time_std = np.sqrt(self.lead_time) * residual_std
        df['safety_stock'] = np.ceil(df['z_score'] * lead_time_std)
        
        # 3. Calculate Lead Time Demand (Forecast * Lead Time)
        df['lead_time_demand'] = np.ceil(df['ensemble_forecast'] * self.lead_time)
        
        # 4. Calculate Reorder Point (ROP) and Recommended Order
        df['reorder_point'] = df['lead_time_demand'] + df['safety_stock']
        
        # For this daily system, we recommend ordering up to the ROP
        df['recommended_order_quantity'] = np.maximum(df['reorder_point'], 0)
        
        return df[['date', 'store_id', 'product_id', 'base_price', 'ensemble_forecast', 
                   'target_service_level', 'safety_stock', 'reorder_point', 'recommended_order_quantity']]

if __name__ == "__main__":
    print("Optimization module ready.")