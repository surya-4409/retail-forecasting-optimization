import pandas as pd
import numpy as np
import yaml
import os

class FeatureEngineer:
    def __init__(self, config_path='../config.yaml'):
        self.cat_mappings = {}
        
        # Load configuration
        if not os.path.exists(config_path):
            config_path = 'config.yaml'
            
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def create_date_features(self, df, date_col='date'):
        df[date_col] = pd.to_datetime(df[date_col])
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['quarter'] = df[date_col].dt.quarter
        df['month'] = df[date_col].dt.month
        df['year'] = df[date_col].dt.year
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_holiday'] = df[date_col].apply(lambda x: 1 if (x.month == 1 and x.day == 1) or (x.month == 12 and x.day == 25) else 0)
        return df

    def create_lag_features(self, df, target_col='imputed_sales', lags=None):
        # Dynamically pull from config
        lags = lags or self.config.get('features', {}).get('lags', [1, 7, 14, 28]) 
        df = df.sort_values(by=['store_id', 'product_id', 'date'])
        
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby(['store_id', 'product_id'])[target_col].shift(lag)
        return df

    def create_rolling_features(self, df, target_col='imputed_sales', windows=None):
        # Dynamically pull from config
        windows = windows or self.config.get('features', {}).get('rolling_windows', [7, 14, 28])
        df = df.sort_values(by=['store_id', 'product_id', 'date'])
        
        for window in windows:
            grouped = df.groupby(['store_id', 'product_id'])[target_col]
            df[f'rolling_mean_{window}'] = grouped.transform(lambda x: x.shift(1).rolling(window=window).mean())
            df[f'rolling_std_{window}'] = grouped.transform(lambda x: x.shift(1).rolling(window=window).std())
        return df

    def encode_categorical(self, df, cols=['store_id', 'product_id', 'category', 'store_type', 'location_tier', 'demand_type']):
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
                df[f'{col}_encoded'] = df[col].cat.codes
                self.cat_mappings[col] = dict(enumerate(df[col].cat.categories))
        return df

    def run_pipeline(self, df):
        df = self.create_date_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.encode_categorical(df)
        df = df.dropna()
        return df