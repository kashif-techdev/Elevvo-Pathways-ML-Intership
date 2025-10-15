"""
Sales Forecasting Project with Mandatory Bonus Features
Walmart Sales Forecast Implementation

This script implements a comprehensive sales forecasting system with:
- Time-based feature engineering
- Lag features and rolling averages (bonus)
- Seasonal decomposition (bonus)
- Multiple ML models including XGBoost and LightGBM (bonus)
- Time-aware cross-validation (bonus)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose

class SalesForecaster:
    """
    Comprehensive Sales Forecasting System
    """
    
    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.scaler = None
        
    def load_and_prepare_data(self, train_path, features_path, stores_path):
        """
        Step 1 & 2: Load and preprocess Walmart dataset
        """
        print("Loading and preprocessing data...")
        
        # Load datasets
        train_df = pd.read_csv(train_path)
        features_df = pd.read_csv(features_path)
        stores_df = pd.read_csv(stores_path)
        
        print(f"Train data shape: {train_df.shape}")
        print(f"Features data shape: {features_df.shape}")
        print(f"Stores data shape: {stores_df.shape}")
        
        # Merge datasets
        # First merge train with features
        df = train_df.merge(features_df, on=['Store', 'Date'], how='left')
        # Then merge with stores
        df = df.merge(stores_df, on='Store', how='left')
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by Store, Dept, and Date
        df = df.sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        print(f"Final dataset shape: {df.shape}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        
        # Check for missing values
        missing_info = df.isnull().sum()
        print("Missing values per column:")
        print(missing_info[missing_info > 0])
        
        # Fill missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def create_features(self, df):
        """
        Step 3: Feature Engineering (Core + Bonus Features)
        """
        print("Creating features...")
        
        # Make a copy to avoid modifying original
        df_features = df.copy()
        
        # Time-based features
        df_features['year'] = df_features['Date'].dt.year
        df_features['month'] = df_features['Date'].dt.month
        df_features['week'] = df_features['Date'].dt.isocalendar().week
        df_features['dayofweek'] = df_features['Date'].dt.dayofweek
        df_features['is_weekend'] = df_features['dayofweek'].isin([5, 6]).astype(int)
        df_features['quarter'] = df_features['Date'].dt.quarter
        
        # Lag features (previous sales)
        df_features['lag_1'] = df_features.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
        df_features['lag_2'] = df_features.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(2)
        df_features['lag_3'] = df_features.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(3)
        df_features['lag_4'] = df_features.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(4)
        
        # Rolling averages (BONUS FEATURE)
        df_features['rolling_mean_3'] = df_features.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(window=3).mean()
        df_features['rolling_mean_7'] = df_features.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(window=7).mean()
        df_features['rolling_std_3'] = df_features.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(window=3).std()
        df_features['rolling_std_7'] = df_features.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(window=7).std()
        
        # Rolling min/max
        df_features['rolling_min_3'] = df_features.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(window=3).min()
        df_features['rolling_max_3'] = df_features.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(window=3).max()
        
        # Seasonal decomposition (BONUS FEATURE)
        print("Performing seasonal decomposition...")
        df_features = self._add_seasonal_features(df_features)
        
        # Additional features
        df_features['sales_ratio'] = df_features['Weekly_Sales'] / df_features.groupby(['Store', 'Dept'])['Weekly_Sales'].transform('mean')
        df_features['price_ratio'] = df_features['CPI'] / df_features.groupby(['Store'])['CPI'].transform('mean')
        
        # Holiday features
        df_features['is_holiday'] = df_features['IsHoliday'].astype(int)
        df_features['days_to_holiday'] = self._calculate_days_to_holiday(df_features)
        
        # Store and department features
        df_features['store_size_category'] = pd.cut(df_features['Size'], bins=3, labels=['Small', 'Medium', 'Large'])
        df_features['store_size_category'] = df_features['store_size_category'].astype('category').cat.codes
        
        # Type encoding
        df_features['type_encoded'] = df_features['Type'].astype('category').cat.codes
        
        # Remove rows with NaN values created by lag features
        df_features = df_features.dropna()
        
        print(f"Features created. Final shape: {df_features.shape}")
        
        return df_features
    
    def _add_seasonal_features(self, df):
        """Add seasonal decomposition features (BONUS)"""
        seasonal_features = []
        
        # Apply seasonal decomposition for each store-dept combination
        for (store, dept), group in df.groupby(['Store', 'Dept']):
            if len(group) >= 52:  # Need at least 52 weeks for yearly seasonality
                try:
                    # Perform seasonal decomposition
                    result = seasonal_decompose(group['Weekly_Sales'], model='additive', period=52)
                    
                    # Add to dataframe
                    group['trend'] = result.trend
                    group['seasonal'] = result.seasonal
                    group['residual'] = result.resid
                    
                    seasonal_features.append(group)
                except:
                    # If decomposition fails, add zeros
                    group['trend'] = 0
                    group['seasonal'] = 0
                    group['residual'] = 0
                    seasonal_features.append(group)
            else:
                # Not enough data for seasonal decomposition
                group['trend'] = 0
                group['seasonal'] = 0
                group['residual'] = 0
                seasonal_features.append(group)
        
        return pd.concat(seasonal_features, ignore_index=True)
    
    def _calculate_days_to_holiday(self, df):
        """Calculate days to next holiday"""
        df = df.copy()
        df['days_to_holiday'] = 0
        
        # Find holiday dates
        holiday_dates = df[df['IsHoliday'] == True]['Date'].unique()
        
        for idx, row in df.iterrows():
            if row['IsHoliday']:
                df.loc[idx, 'days_to_holiday'] = 0
            else:
                # Find next holiday
                future_holidays = holiday_dates[holiday_dates > row['Date']]
                if len(future_holidays) > 0:
                    next_holiday = min(future_holidays)
                    days_to_holiday = (next_holiday - row['Date']).days
                    df.loc[idx, 'days_to_holiday'] = days_to_holiday
                else:
                    df.loc[idx, 'days_to_holiday'] = 365  # Default to 1 year
        
        return df['days_to_holiday']
    
    def prepare_modeling_data(self, df, target_col='Weekly_Sales'):
        """
        Step 4: Prepare data for modeling with time-aware splitting
        """
        print("Preparing modeling data...")
        
        # Select features for modeling
        feature_cols = [col for col in df.columns if col not in [
            'Date', 'Weekly_Sales', 'Store', 'Dept', 'Type', 'IsHoliday'
        ]]
        
        self.feature_columns = feature_cols
        X = df[feature_cols]
        y = df[target_col]
        
        print(f"Features used: {len(feature_cols)}")
        print(f"Feature columns: {feature_cols}")
        
        # Time-aware split (80-20)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, df[split_idx:]['Date']
    
    def train_baseline_models(self, X_train, y_train):
        """
        Step 5: Train baseline models
        """
        print("Training baseline models...")
        
        # Linear Regression
        print("Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        self.models['LinearRegression'] = lr_model
        
        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['RandomForest'] = rf_model
        
        print("Baseline models trained!")
    
    def train_advanced_models(self, X_train, y_train):
        """
        Step 6: Train advanced models (BONUS FEATURES)
        """
        print("Training advanced models...")
        
        # XGBoost (BONUS)
        print("Training XGBoost...")
        xgb_model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_model
        
        # LightGBM (BONUS)
        print("Training LightGBM...")
        lgb_model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        self.models['LightGBM'] = lgb_model
        
        print("Advanced models trained!")
    
    def time_aware_cross_validation(self, X_train, y_train, model_name='XGBoost'):
        """
        Time-aware cross-validation (BONUS FEATURE)
        """
        print(f"Performing time-aware cross-validation for {model_name}...")
        
        # Initialize model
        if model_name == 'XGBoost':
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        elif model_name == 'LightGBM':
            model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            print(f"  Fold {fold + 1}/5")
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train model
            model.fit(X_tr, y_tr)
            
            # Predict and evaluate
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            cv_scores.append(mae)
            
            print(f"    MAE: {mae:.2f}")
        
        avg_mae = np.mean(cv_scores)
        print(f"Average MAE across folds: {avg_mae:.2f}")
        
        return cv_scores
    
    def evaluate_models(self, X_test, y_test, test_dates):
        """
        Step 7: Evaluate all models and create visualizations
        """
        print("Evaluating models...")
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'predictions': y_pred
            }
            
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²: {r2:.4f}")
        
        # Create visualizations
        self._create_visualizations(y_test, results, test_dates)
        
        return results
    
    def _create_visualizations(self, y_test, results, test_dates):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Sales Forecasting Results', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted for all models
        ax1 = axes[0, 0]
        ax1.plot(test_dates, y_test, label='Actual', color='blue', linewidth=2)
        
        colors = ['red', 'green', 'orange', 'purple']
        for i, (model_name, result) in enumerate(results.items()):
            ax1.plot(test_dates, result['predictions'], 
                    label=f'{model_name} (MAE: {result["MAE"]:.0f})', 
                    color=colors[i], alpha=0.7, linewidth=1.5)
        
        ax1.set_title('Actual vs Predicted Sales')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Weekly Sales')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Model comparison bar chart
        ax2 = axes[0, 1]
        model_names = list(results.keys())
        mae_scores = [results[name]['MAE'] for name in model_names]
        rmse_scores = [results[name]['RMSE'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax2.bar(x - width/2, mae_scores, width, label='MAE', alpha=0.8)
        ax2.bar(x + width/2, rmse_scores, width, label='RMSE', alpha=0.8)
        
        ax2.set_title('Model Performance Comparison')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Error')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals plot for best model
        best_model = min(results.keys(), key=lambda x: results[x]['MAE'])
        ax3 = axes[1, 0]
        residuals = y_test - results[best_model]['predictions']
        ax3.scatter(results[best_model]['predictions'], residuals, alpha=0.6)
        ax3.axhline(y=0, color='red', linestyle='--')
        ax3.set_title(f'Residuals Plot - {best_model}')
        ax3.set_xlabel('Predicted Values')
        ax3.set_ylabel('Residuals')
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature importance (if available)
        ax4 = axes[1, 1]
        if hasattr(self.models[best_model], 'feature_importances_'):
            importances = self.models[best_model].feature_importances_
            feature_names = self.feature_columns
            
            # Get top 10 features
            top_indices = np.argsort(importances)[-10:]
            top_features = [feature_names[i] for i in top_indices]
            top_importances = importances[top_indices]
            
            ax4.barh(range(len(top_features)), top_importances)
            ax4.set_yticks(range(len(top_features)))
            ax4.set_yticklabels(top_features)
            ax4.set_title(f'Top 10 Feature Importance - {best_model}')
            ax4.set_xlabel('Importance')
        else:
            ax4.text(0.5, 0.5, f'{best_model}\nFeature importance\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title(f'Feature Importance - {best_model}')
        
        plt.tight_layout()
        plt.savefig('sales_forecasting_results.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'sales_forecasting_results.png'")
        # plt.show()  # Commented out to avoid display issues in terminal
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        for model_name, result in results.items():
            print(f"{model_name:15} | MAE: {result['MAE']:8.2f} | RMSE: {result['RMSE']:8.2f} | R²: {result['R2']:6.4f}")
        
        best_model = min(results.keys(), key=lambda x: results[x]['MAE'])
        print(f"\nBest Model: {best_model} (Lowest MAE: {results[best_model]['MAE']:.2f})")
        print("="*60)

def main():
    """
    Main execution function
    """
    print("Starting Sales Forecasting Project")
    print("="*50)
    
    # Initialize forecaster
    forecaster = SalesForecaster()
    
    # Note: In a real scenario, you would download the actual Walmart dataset
    # For this implementation, we'll create a synthetic dataset that mimics the structure
    print("Creating synthetic Walmart-like dataset...")
    
    # Create synthetic data
    np.random.seed(42)
    n_stores = 5
    n_depts = 3
    n_weeks = 150
    
    data = []
    for store in range(1, n_stores + 1):
        for dept in range(1, n_depts + 1):
            for week in range(n_weeks):
                date = pd.date_range('2020-01-01', periods=n_weeks, freq='W')[week]
                
                # Create realistic sales pattern with seasonality
                base_sales = 10000 + store * 2000 + dept * 1000
                seasonal = 2000 * np.sin(2 * np.pi * week / 52)  # Yearly seasonality
                trend = week * 10  # Slight upward trend
                noise = np.random.normal(0, 1000)
                
                weekly_sales = max(0, base_sales + seasonal + trend + noise)
                
                # Add holiday effects
                is_holiday = (week % 52 in [45, 46, 47, 48]) or (week % 52 in [0, 1, 2, 3])
                if is_holiday:
                    weekly_sales *= 1.3
                
                data.append({
                    'Store': store,
                    'Dept': dept,
                    'Date': date,
                    'Weekly_Sales': weekly_sales,
                    'IsHoliday': is_holiday,
                    'Size': np.random.choice([10000, 20000, 30000]),
                    'Type': np.random.choice(['A', 'B', 'C']),
                    'Temperature': np.random.normal(70, 20),
                    'Fuel_Price': np.random.normal(3.5, 0.5),
                    'CPI': np.random.normal(200, 20),
                    'Unemployment': np.random.normal(8, 2)
                })
    
    df = pd.DataFrame(data)
    print(f"Synthetic dataset created: {df.shape}")
    
    # Step 1-2: Data preprocessing
    df_processed = forecaster.create_features(df)
    
    # Step 4: Prepare modeling data
    X_train, X_test, y_train, y_test, test_dates = forecaster.prepare_modeling_data(df_processed)
    
    # Step 5: Train baseline models
    forecaster.train_baseline_models(X_train, y_train)
    
    # Step 6: Train advanced models
    forecaster.train_advanced_models(X_train, y_train)
    
    # Time-aware cross-validation (BONUS)
    print("\nTime-aware Cross-Validation Results:")
    forecaster.time_aware_cross_validation(X_train, y_train, 'XGBoost')
    forecaster.time_aware_cross_validation(X_train, y_train, 'LightGBM')
    
    # Step 7: Evaluate and visualize
    results = forecaster.evaluate_models(X_test, y_test, test_dates)
    
    print("\nSales Forecasting Project Completed!")
    print("All mandatory bonus features implemented:")
    print("   - Rolling averages and statistical features")
    print("   - Seasonal decomposition")
    print("   - XGBoost and LightGBM models")
    print("   - Time-aware cross-validation")
    print("   - Comprehensive visualizations")

if __name__ == "__main__":
    main()
