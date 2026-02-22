import pandas as pd
import numpy as np
from datetime import timedelta
import os

def generate_retail_data(base_path="data", num_stores=5, num_products=20, start_date="2024-01-01", days=730):
    """Generates synthetic retail data including stores, products, promotions, and daily sales."""
    
    np.random.seed(42) # For reproducibility
    os.makedirs(base_path, exist_ok=True)
    
    # 1. Generate Store Data
    stores = pd.DataFrame({
        'store_id': [f'S{str(i).zfill(3)}' for i in range(1, num_stores + 1)],
        'store_type': np.random.choice(['Flagship', 'Standard', 'Express'], num_stores),
        'location_tier': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], num_stores)
    })
    stores.to_csv(f"{base_path}/stores.csv", index=False)
    
    # 2. Generate Product Hierarchy Data
    categories = ['Electronics', 'Clothing', 'Groceries']
    products = pd.DataFrame({
        'product_id': [f'P{str(i).zfill(3)}' for i in range(1, num_products + 1)],
        'category': np.random.choice(categories, num_products),
        'base_price': np.random.uniform(10, 500, num_products).round(2),
        # Segmenting products as required (Fast-moving vs Intermittent)
        'demand_type': np.random.choice(['Fast-Moving', 'Intermittent', 'Seasonal'], num_products, p=[0.5, 0.3, 0.2])
    })
    products.to_csv(f"{base_path}/products.csv", index=False)
    
    # 3. Generate Promotional Calendar
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    promotions = pd.DataFrame({'date': dates})
    # Add some random promotional periods
    promotions['is_promo'] = np.random.choice([0, 1], size=days, p=[0.9, 0.1])
    promotions['discount_depth'] = promotions['is_promo'] * np.random.uniform(0.1, 0.5, days).round(2)
    promotions.to_csv(f"{base_path}/promotions.csv", index=False)
    
    # 4. Generate Sales Data (The Core Time-Series)
    sales_records = []
    
    for _, store in stores.iterrows():
        store_multiplier = 1.5 if store['store_type'] == 'Flagship' else (0.8 if store['store_type'] == 'Express' else 1.0)
        
        for _, product in products.iterrows():
            # Base daily demand
            if product['demand_type'] == 'Fast-Moving':
                base_demand = np.random.poisson(lam=50)
            elif product['demand_type'] == 'Intermittent':
                base_demand = np.random.poisson(lam=2) # Lots of zeros
            else: # Seasonal
                base_demand = np.random.poisson(lam=20)
                
            for i, date in enumerate(dates):
                # Calculate daily demand with seasonality, trend, and promo lift
                day_of_year = date.timetuple().tm_yday
                
                # Annual seasonality sine wave
                seasonality = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365) if product['demand_type'] == 'Seasonal' else 1.0
                
                # Slight upward trend over time
                trend = 1 + (i / days) * 0.2 
                
                # Promo lift
                is_promo = promotions.loc[promotions['date'] == date, 'is_promo'].values[0]
                promo_lift = 1 + (is_promo * 1.5) 
                
                # Final demand calculation with some noise
                daily_demand = max(0, int(base_demand * store_multiplier * seasonality * trend * promo_lift * np.random.normal(1, 0.1)))
                
                # Introduce occasional stockouts (missing data/zero sales despite demand)
                is_stockout = np.random.choice([True, False], p=[0.02, 0.98]) # 2% chance of stockout
                if is_stockout:
                    daily_demand = 0 
                
                sales_records.append({
                    'date': date,
                    'store_id': store['store_id'],
                    'product_id': product['product_id'],
                    'sales_quantity': daily_demand,
                    'is_stockout': int(is_stockout)
                })

    sales = pd.DataFrame(sales_records)
    sales.to_csv(f"{base_path}/sales.csv", index=False)
    print(f"Data generation complete! Files saved to {base_path}")

if __name__ == "__main__":
    generate_retail_data()