import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Suppress statsmodels warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

class ForecastingModels:
    def __init__(self):
        self.xgb_model = None
        self.residual_std = 0 # To store uncertainty for prediction intervals
        self.features = [
            'base_price', 'is_promo', 'discount_depth', 'day_of_week', 'quarter', 
            'month', 'year', 'day_of_year', 'is_weekend', 'is_holiday', 'lag_1', 
            'lag_7', 'lag_14', 'lag_28', 'rolling_mean_7', 'rolling_std_7', 
            'rolling_mean_14', 'rolling_std_14', 'rolling_mean_28', 'rolling_std_28',
            'store_id_encoded', 'product_id_encoded', 'category_encoded', 
            'store_type_encoded', 'location_tier_encoded', 'demand_type_encoded'
        ]
        self.target = 'imputed_sales'

    def walk_forward_split(self, df, n_splits=3):
        """Generates train/test splits for robust Time-Series Cross-Validation."""
        # Get unique sorted dates
        dates = np.sort(df['date'].unique())
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        splits = []
        for train_idx, test_idx in tscv.split(dates):
            train_dates = dates[train_idx]
            test_dates = dates[test_idx]
            
            train_df = df[df['date'].isin(train_dates)].copy()
            test_df = df[df['date'].isin(test_dates)].copy()
            splits.append((train_df, test_df))
            
        return splits

    def train_xgboost(self, train_df):
        """Trains XGBoost and calculates training residuals for uncertainty intervals."""
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
        
        # Calculate residuals on training data to quantify uncertainty
        train_preds = self.xgb_model.predict(X_train)
        self.residual_std = np.std(y_train - train_preds)
        
        return self.xgb_model

    def predict_xgboost(self, test_df, confidence_level=1.96):
        """Generates point predictions and prediction intervals."""
        X_test = test_df[self.features]
        preds = self.xgb_model.predict(X_test)
        preds = np.maximum(preds, 0) # Sales can't be negative
        
        # 95% Confidence Interval bounds based on residual std
        lower_bound = np.maximum(preds - (confidence_level * self.residual_std), 0)
        upper_bound = preds + (confidence_level * self.residual_std)
        
        return preds, lower_bound, upper_bound

    def croston_forecast(self, ts, test_steps):
        """Implementation of Croston's method for Intermittent demand."""
        ts_array = np.array(ts)
        non_zero_idx = np.where(ts_array > 0)[0]
        
        if len(non_zero_idx) == 0:
            return np.zeros(test_steps)
            
        demands = ts_array[non_zero_idx]
        intervals = np.diff(np.insert(non_zero_idx, 0, -1))
        
        # Average demand size and average interval between demands
        mean_demand = np.mean(demands)
        mean_interval = np.mean(intervals)
        
        forecast = mean_demand / mean_interval if mean_interval > 0 else 0
        return np.full(test_steps, forecast)

    def train_predict_statistical(self, train_series, test_steps, demand_type, seasonal_periods=7):
        """Routes to the correct statistical baseline based on product segmentation."""
        if demand_type == 'Intermittent':
            # Use Croston's for sparse data
            preds = self.croston_forecast(train_series, test_steps)
        else:
            # Use Exponential Smoothing for Fast-Moving and Seasonal
            try:
                model = ExponentialSmoothing(
                    train_series + 0.001, 
                    trend='add', 
                    seasonal='add', 
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated"
                )
                fit_model = model.fit()
                preds = fit_model.forecast(test_steps).values
            except:
                # Fallback to simple mean if statsmodels fails to converge
                preds = np.full(test_steps, train_series.mean())
                
        return np.maximum(preds, 0)

    def calculate_metrics(self, y_true, y_pred):
        """Calculates required business metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        # Add epsilon to prevent division by zero for MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
        return {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE (%)': round(mape, 2)}