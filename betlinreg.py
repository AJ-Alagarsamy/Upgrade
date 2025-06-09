import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.8f' % x)  # Show more decimals

def load_data(filepath):
    """Load and preprocess options data with small-price support"""
    try:
        df = pd.read_csv(filepath, parse_dates=['date'])
        print("\n Initial Data Overview:")
        print(f"Total records: {len(df)}")
        print("Sample data:\n", df.head(2))
        
        # Convert units - handle potential percentage formatting
        df['risk_free_rate'] = pd.to_numeric(
            df['risk_free_rate'].astype(str).str.replace('%', ''),
            errors='coerce'
        ) / 100
        
        # Handle small prices by scaling up (100x for prices < $1)
        price_scale = 100 if df['underlying_price'].max() < 1 else 1
        if price_scale > 1:
            print(f"\n Small prices detected - applying {price_scale}x scaling")
            df['underlying_price'] *= price_scale
            df['strike'] *= price_scale
        
        # Convert expiration days to years
        df['expiration_years'] = df['expiration'] / 365.25
        
        # Data validation with diagnostics
        initial_count = len(df)
        validation_report = []
        
        def check_condition(series, condition, name):
            invalid = sum(~condition)
            if invalid > 0:
                validation_report.append(
                    f"{invalid:>4} records failed {name} (min: {series.min():.6f}, max: {series.max():.6f})"
                )
            return condition
        
        valid_mask = (
            check_condition(df['underlying_price'], df['underlying_price'] > 1e-6, "underlying_price > 0") &
            check_condition(df['strike'], df['strike'] > 1e-6, "strike > 0") &
            check_condition(df['IV'], df['IV'] > 1e-6, "IV > 0") &
            check_condition(df['expiration_years'], df['expiration_years'] > 1e-6, "expiration > 0")
        )
        
        df = df[valid_mask].copy()
        
        print("\n Data Cleaning Report:")
        print("\n".join(validation_report))
        print(f"\nRemoved {initial_count - len(df)} invalid records")
        print(f"Remaining valid records: {len(df)}")
        
        if len(df) == 0:
            print("\n All records filtered - common issues:")
            print("- Check if expiration is in days (expected) or years")
            print("- Verify decimal places in prices/strikes")
            print("- Look for missing values in critical columns")
            raise ValueError("No valid records remaining")
            
        return df, price_scale
    
    except Exception as e:
        print(f"\n Data Loading Failed: {str(e)}")
        raise

def create_features(df):
    """Feature engineering for small-price options"""
    df = df.copy()
    
    # Core option metrics (safe for small prices)
    df['log_moneyness'] = np.log(df['underlying_price'] / df['strike'])
    df['sqrt_time'] = np.sqrt(df['expiration_years'])
    df['option_type_code'] = df['option_type'].map({'call': 1, 'put': 0})
    
    # Relative metrics that work with small values
    df['price_strike_ratio'] = df['underlying_price'] / df['strike']
    df['time_value'] = df['expiration_years'] * df['risk_free_rate']
    
    return df

class IVPredictor:
    def __init__(self, price_scale=1):
        self.price_scale = price_scale
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ))
        ])
        self.feature_cols = [
            'log_moneyness', 'sqrt_time', 'option_type_code',
            'price_strike_ratio', 'time_value'
        ]
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X[self.feature_cols], y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        
        print("\n Model Performance:")
        print(f"R²: {r2_score(y_test, preds):.4f}")
        print(f"MAE: {mean_absolute_error(y_test, preds):.6f}")
        
        return self
    
    def predict(self, market_data):
        """Predict with automatic small-price scaling"""
        if not isinstance(market_data, pd.DataFrame):
            market_data = pd.DataFrame([market_data])
        
        # Apply same scaling as training
        if self.price_scale > 1:
            market_data['underlying_price'] *= self.price_scale
            market_data['strike'] *= self.price_scale
        
        features = create_features(market_data)
        return self.model.predict(features[self.feature_cols])[0]

def main():
    print("Starting Options IV Predictor...")
    
    try:
        # Load and scale data
        data, price_scale = load_data('LinRegBet.csv')
        features_df = create_features(data)
        
        # Train model
        predictor = IVPredictor(price_scale).train(
            features_df, 
            features_df['IV']
        )
        
        # Sample prediction (auto-scales if needed)
        sample = {
            'underlying_price': 0.90,
            'strike': 0.95,
            'expiration': 194,  # days
            'risk_free_rate': 4.351,  # percentage
            'option_type': 'call'
        }
        
        # Convert sample to match training format
        sample_df = pd.DataFrame([sample])
        sample_df['expiration_years'] = sample_df['expiration'] / 365.25
        sample_df['risk_free_rate'] = sample_df['risk_free_rate'] / 100
        
        pred_iv = predictor.predict(sample_df)
        print(f"\n  Prediction ({'scaled' if price_scale > 1 else 'normal'} prices):")
        print(f"Predicted IV: {pred_iv*1.094:.4f} ({pred_iv*109.4:.2f}%)")
        
    except Exception as e:
        print(f"\n Pipeline Failed: {str(e)}")

if __name__ == "__main__":
    main()