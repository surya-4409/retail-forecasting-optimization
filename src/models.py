import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class ForecastingModels:
    def __init__(self):
        self.xgb_model = None
        self.residual_std = 0  # NEW: We will store model uncertainty here
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
        
    def walk_forward_split(self, df, n_splits=3, test_length=30):
        """Generates splits for walk-forward cross validation."""
        splits = []
        max_date = df['date'].max()
        
        for i in range(n_splits, 0, -1):
            test_end = max_date - pd.Timedelta(days=(i-1)*test_length)
            test_start = test_end - pd.Timedelta(days=test_length)
            
            train = df[df['date'] < test_start].copy()
            test = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
            splits.append((train, test))
        return splits

    def train_xgboost(self, train_df):
        """Trains the global XGBoost model on all data and calculates uncertainty."""
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
        
        # Calculate and store the residual standard deviation for prediction intervals
        train_preds = np.maximum(self.xgb_model.predict(X_train), 0)
        self.residual_std = np.std(y_train - train_preds)
        
        return self.xgb_model

    def predict_xgboost(self, test_df):
        """Generates ML predictions and confidence bounds."""
        X_test = test_df[self.features]
        preds = np.maximum(self.xgb_model.predict(X_test), 0)
        
        # Calculate 95% confidence intervals using the saved residual_std
        lower_bound = np.maximum(preds - (1.96 * self.residual_std), 0)
        upper_bound = preds + (1.96 * self.residual_std)
        
        # Return exactly the 3 items the notebook expects
        return preds, lower_bound, upper_bound

    def train_predict_expsmoothing(self, train_series, test_steps, seasonal_periods=7):
        """Statistical Baseline: Exponential Smoothing for a single series."""
        model = ExponentialSmoothing(
            train_series + 0.001, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=seasonal_periods,
            initialization_method="estimated"
        )
        fit_model = model.fit()
        return np.maximum(fit_model.forecast(test_steps).values, 0)
        
    def train_predict_statistical(self, train_series, test_steps, demand_type):
        """Routes the series to the correct statistical model based on demand segment."""
        if demand_type == 'Intermittent':
            mean_demand = train_series[train_series > 0].mean() if (train_series > 0).any() else 0
            prob_demand = (train_series > 0).mean()
            return np.full(test_steps, mean_demand * prob_demand)
        else:
            try:
                return self.train_predict_expsmoothing(train_series, test_steps)
            except:
                return np.full(test_steps, train_series.mean())

    def generate_ensemble(self, xgb_preds, es_preds, weight_xgb=0.5, weight_es=0.5):
        """Explicitly merges ML and statistical forecasts using weighted averaging."""
        if len(xgb_preds) != len(es_preds):
            raise ValueError("Prediction arrays must be the same length to ensemble.")
        return (xgb_preds * weight_xgb) + (es_preds * weight_es)

    def calculate_metrics(self, y_true, y_pred):
        """Calculates required business metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
        return {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE (%)': round(mape, 2)}