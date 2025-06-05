import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import newton
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 1. Black-Scholes calculation functions (optimized)
def black_scholes_price(S, K, T, r, sigma, option_type):
    """Vectorized Black-Scholes calculation"""
    if T <= 0 or sigma <= 0:
        return np.maximum(0, S - K) if option_type == 'call' else np.maximum(0, K - S)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility(S, K, T, r, option_price, option_type, max_iter=100, precision=1e-6):
    """Robust IV calculation with multiple fallbacks"""
    if T <= 0 or option_price <= 0:
        return 0.01
    
    # Initial bounds
    sigma_low, sigma_high = 0.001, 5.0
    
    try:
        # Newton-Raphson with fallback to bisection
        f = lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - option_price
        iv = newton(f, 0.3, maxiter=max_iter, tol=precision)
        return np.clip(iv, 0.01, 5.0)
    except:
        # Binary search fallback
        for _ in range(max_iter):
            sigma_mid = (sigma_low + sigma_high) / 2
            price_mid = black_scholes_price(S, K, T, r, sigma_mid, option_type)
            
            if abs(price_mid - option_price) < precision:
                return sigma_mid
            elif price_mid < option_price:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid
        return (sigma_low + sigma_high) / 2

# 2. Data loading and preprocessing
def load_and_preprocess_data(filepath):
    """Load and preprocess the data file"""
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Convert dates and calculate time to expiration (in years)
    df['date'] = pd.to_datetime(df['date'])
    df['expiration_years'] = df['expiration'] / 365  # Convert days to years
    
    # Convert risk-free rate from percentage to decimal
    df['risk_free_rate'] = df['risk_free_rate'] / 100
    
    # Clean data - less strict filtering
    df = df.dropna()
    
    # Instead of filtering out rows, we'll fill or adjust problematic values
    df['underlying_price'] = df['underlying_price'].replace(0, np.nan).fillna(method='ffill')
    df['historical_vol'] = df['historical_vol'].replace(0, np.nan).fillna(method='ffill')
    
    # For this example, we'll use the provided IV column instead of calculating it
    # since we don't have actual option prices in the CSV
    df['IV_calculated'] = df['IV']
    
    return df

# 3. Feature engineering
def create_features(df):
    """Create advanced features for the model"""
    # Basic features
    df['log_moneyness'] = np.log(df['underlying_price'] / df['strike'])
    df['sqrt_time'] = np.sqrt(df['expiration_years'])
    df['option_type_code'] = df['option_type'].map({'call': 1, 'put': 0})
    
    # Advanced features
    df['vol_spread'] = df['IV'] - df['historical_vol']
    df['moneyness_sq'] = (df['underlying_price'] / df['strike'])**2
    df['time_vol_interaction'] = df['sqrt_time'] * df['historical_vol']
    
    return df

# 4. Model training with feature selection
def train_iv_model(X, y):
    """Train optimized polynomial regression model"""
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression()
    )
    
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    print(f"Model Performance:")
    print(f"- R²: {r2_score(y, y_pred):.4f}")
    print(f"- MAE: {mean_absolute_error(y, y_pred):.4f}")
    
    # Feature importance
    print("\nFeature Importance:")
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    for i in result.importances_mean.argsort()[::-1]:
        print(f"{X.columns[i]:<25}: {result.importances_mean[i]:.3f}")
    
    return model

# 5. Prediction function for live data
def predict_iv(model, market_params, feature_cols):
    """Predict IV for current market conditions"""
    features = pd.DataFrame([{
        'log_moneyness': np.log(market_params['underlying_price'] / market_params['strike']),
        'sqrt_time': np.sqrt(market_params['expiration_days'] / 365),
        'historical_vol': market_params['hist_vol'],
        'risk_free_rate': market_params['risk_free_rate'] / 100,  # Convert percentage to decimal
        'option_type_code': 1 if market_params['option_type'] == 'call' else 0,
        'vol_spread': market_params.get('vol_spread', 0),
        'moneyness_sq': (market_params['underlying_price'] / market_params['strike'])**2,
        'time_vol_interaction': np.sqrt(market_params['expiration_days']/365) * market_params['hist_vol']
    }])
    
    # Select only the features the model was trained on
    features = features[feature_cols]
    return model.predict(features)[0]

# Main execution
if __name__ == "__main__":
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        data = load_and_preprocess_data('LinRegBet.csv')
        
        if data.empty:
            raise ValueError("No data available after preprocessing. Check your input file.")
        
        print(f"Loaded {len(data)} rows of data")
        print("First few rows:")
        print(data.head())
        
        data = create_features(data)
        
        # Define features and target
        feature_cols = [
            'log_moneyness', 'sqrt_time', 'historical_vol', 
            'risk_free_rate', 'option_type_code', 'vol_spread',
            'moneyness_sq', 'time_vol_interaction'
        ]
        
        # Ensure all feature columns exist
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        X_train = data[feature_cols]
        y_train = data['IV']  # Using the provided IV from the CSV as target
        
        if len(X_train) == 0:
            raise ValueError("Training data is empty after preprocessing.")
        
        # Train model
        print("\nTraining model...")
        iv_model = train_iv_model(X_train, y_train)
        
        # Example prediction using the last row's parameters
        last_row = data.iloc[-1]
        current_market = {
            'underlying_price': last_row['underlying_price'],
            'strike': last_row['strike'],
            'expiration_days': last_row['expiration'] * 365,  # Convert back to days
            'hist_vol': last_row['historical_vol'],
            'risk_free_rate': last_row['risk_free_rate'] * 100,  # Convert back to percentage
            'option_type': last_row['option_type'],
            'vol_spread': last_row['vol_spread']
        }
        
        # Predict IV
        predicted_iv = predict_iv(iv_model, current_market, feature_cols)
        print(f"\nPredicted Implied Volatility: {predicted_iv:.4f} ({predicted_iv*100:.1f}%)")
        
        # Compare with the actual IV from the CSV
        actual_iv = last_row['IV']
        print(f"Actual IV from CSV: {actual_iv:.4f} ({actual_iv*100:.1f}%)")
        print(f"Difference: {(predicted_iv - actual_iv):.4f}")
    
    except Exception as e:
        print(f"Error: {str(e)}")