import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class ForecastingModels:
    def __init__(self):
        self.xgb_model = None
        # These are the numerical and encoded categorical features we created
        self.features = [
            'base_price', 'is_promo', 'discount_depth', 'day_of_week', 'quarter', 
            'month', 'year', 'day_of_year', 'is_weekend', 'is_holiday', 'lag_1', 
            'lag_7', 'lag_14', 'lag_28', 'rolling_mean_7', 'rolling_std_7', 
            'rolling_mean_14', 'rolling_std_14', 'rolling_mean_28', 'rolling_std_28',
            'store_id_encoded', 'product_id_encoded', 'category_encoded', 
            'store_type_encoded', 'location_tier_encoded', 'demand_type_encoded'
        ]
        self.target = 'imputed_sales'

    def chronological_split(self, df, split_date):
        """Splits data chronologically to simulate real-world forecasting."""
        train = df[df['date'] < split_date].copy()
        test = df[df['date'] >= split_date].copy()
        return train, test

    def train_xgboost(self, train_df):
        """Trains the global XGBoost model on all data."""
        X_train = train_df[self.features]
        y_train = train_df[self.target]
        
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=150, 
            learning_rate=0.05, 
            max_depth=6, 
            random_state=42,
            objective='reg:squarederror'
        )
        self.xgb_model.fit(X_train, y_train)
        return self.xgb_model

    def predict_xgboost(self, test_df):
        """Generates ML predictions and calculates prediction intervals based on residual std."""
        X_test = test_df[self.features]
        preds = self.xgb_model.predict(X_test)
        preds = np.maximum(preds, 0) # Sales can't be negative
        return preds

    def train_predict_expsmoothing(self, train_series, test_steps, seasonal_periods=7):
        """Statistical Baseline: Exponential Smoothing for a single series."""
        # Add small constant to avoid division by zero errors in statsmodels
        model = ExponentialSmoothing(
            train_series + 0.001, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=seasonal_periods,
            initialization_method="estimated"
        )
        fit_model = model.fit()
        preds = fit_model.forecast(test_steps)
        return np.maximum(preds.values, 0)

    def calculate_metrics(self, y_true, y_pred):
        """Calculates required business metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        # Add epsilon to prevent division by zero for MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
        return {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE (%)': round(mape, 2)}