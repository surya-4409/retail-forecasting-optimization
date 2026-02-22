import pandas as pd
import numpy as np
import os

class DataPreprocessor:
    def __init__(self, data_dir='../data'):
        # Smart pathing depending on where the script is called from
        if not os.path.exists(data_dir):
            self.data_dir = 'data'
        else:
            self.data_dir = data_dir

    def load_and_merge(self):
        """Loads all CSVs and merges them into a single analytical dataset."""
        sales = pd.read_csv(f'{self.data_dir}/sales.csv', parse_dates=['date'])
        products = pd.read_csv(f'{self.data_dir}/products.csv')
        stores = pd.read_csv(f'{self.data_dir}/stores.csv')
        promotions = pd.read_csv(f'{self.data_dir}/promotions.csv', parse_dates=['date'])

        df = sales.merge(products, on='product_id', how='left')
        df = df.merge(stores, on='store_id', how='left')
        df = df.merge(promotions, on='date', how='left')
        
        # Sort chronologically for time-series validity
        df = df.sort_values(by=['store_id', 'product_id', 'date']).reset_index(drop=True)
        return df

    def handle_stockouts(self, df):
        """Replaces 0 sales with NaN during stockout periods to avoid biasing demand."""
        df['actual_sales'] = df['sales_quantity']
        df.loc[df['is_stockout'] == 1, 'actual_sales'] = np.nan
        return df

    def impute_missing_sales(self, df):
        """Imputes missing demand using linear interpolation per product-store."""
        df['imputed_sales'] = df.groupby(['store_id', 'product_id'])['actual_sales'].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        # Catch any edge-cases at the very beginning of the time series
        df['imputed_sales'] = df['imputed_sales'].bfill().fillna(0)
        return df

    def run_pipeline(self):
        """Executes the full preprocessing pipeline."""
        df = self.load_and_merge()
        df = self.handle_stockouts(df)
        df = self.impute_missing_sales(df)
        return df

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.run_pipeline()
    print(f"Preprocessing complete. Clean dataset shape: {df_clean.shape}")