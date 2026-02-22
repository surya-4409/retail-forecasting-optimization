import pandas as pd
import numpy as np
import os
import yaml
import logging

# Set up professional MLOps logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataPreprocessor:
    def __init__(self, config_path='../config.yaml'):
        # 1. Configuration Management: Load paths from yaml
        if not os.path.exists(config_path):
            config_path = 'config.yaml' # Fallback for terminal execution
            
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.data_dir = self.config['paths']['data_dir']
        if not os.path.exists(self.data_dir):
            self.data_dir = f"../{self.data_dir}"

    def load_and_merge(self):
        """Loads all CSVs and merges them into a single analytical dataset."""
        logging.info("Loading raw CSV files from data directory...")
        sales = pd.read_csv(f'{self.data_dir}/sales.csv', parse_dates=['date'])
        products = pd.read_csv(f'{self.data_dir}/products.csv')
        stores = pd.read_csv(f'{self.data_dir}/stores.csv')
        promotions = pd.read_csv(f'{self.data_dir}/promotions.csv', parse_dates=['date'])

        logging.info("Merging datasets into unified analytical base...")
        df = sales.merge(products, on='product_id', how='left')
        df = df.merge(stores, on='store_id', how='left')
        df = df.merge(promotions, on='date', how='left')
        
        # Sort chronologically for time-series validity
        df = df.sort_values(by=['store_id', 'product_id', 'date']).reset_index(drop=True)
        return df

    def validate_data(self, df):
        """Robust data validation checks (System Design Requirement)."""
        logging.info("Running data validation checks...")
        
        # Check 1: No negative sales quantities
        if (df['sales_quantity'] < 0).any():
            logging.error("Data Validation Failed: Negative sales found in raw data.")
            raise ValueError("Negative sales quantity detected. Halting pipeline.")
            
        # Check 2: No missing critical identifiers
        if df[['store_id', 'product_id', 'date']].isnull().any().any():
            logging.error("Data Validation Failed: Missing primary keys (store, product, or date).")
            raise ValueError("Missing critical identifiers. Halting pipeline.")
            
        logging.info("Data validation passed successfully. Data is clean.")
        return df

    def handle_stockouts(self, df):
        """Replaces 0 sales with NaN during stockout periods to avoid biasing demand."""
        logging.info("Handling historical stockouts...")
        df['actual_sales'] = df['sales_quantity']
        df.loc[df['is_stockout'] == 1, 'actual_sales'] = np.nan
        return df

    def impute_missing_sales(self, df):
        """Imputes missing demand using linear interpolation per product-store."""
        logging.info("Imputing missing sales via linear interpolation...")
        df['imputed_sales'] = df.groupby(['store_id', 'product_id'])['actual_sales'].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        # Catch any edge-cases at the very beginning of the time series
        df['imputed_sales'] = df['imputed_sales'].bfill().fillna(0)
        return df

    def run_pipeline(self):
        """Executes the full preprocessing pipeline."""
        logging.info("Starting Preprocessing Pipeline...")
        df = self.load_and_merge()
        df = self.validate_data(df) # EXECUTING DATA VALIDATION
        df = self.handle_stockouts(df)
        df = self.impute_missing_sales(df)
        logging.info(f"Preprocessing complete. Clean dataset shape: {df.shape}")
        return df

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.run_pipeline()