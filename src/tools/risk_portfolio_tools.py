import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import requests


def calculate_returns_from_prices(prices_data: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """
    Converts price data to daily returns for portfolio analysis.
    
    Args:
        prices_data (dict): Dictionary of coin prices as lists.
                           Example: {"bitcoin": [30000, 31000, 29500, ...], 
                                    "ethereum": [2000, 2100, 1950, ...]}
    
    Returns:
        dict: Daily returns for each coin.
              Example: {"bitcoin": [0.033, -0.048, ...], 
                       "ethereum": [0.05, -0.071, ...]}
        None: If calculation fails.
    """
    try:
        returns_data = {}
        for coin, prices in prices_data.items():
            prices_array = np.array(prices)
            # Calculate returns: (price[i] - price[i-1]) / price[i-1]
            returns = [(prices_array[i] - prices_array[i-1]) / prices_array[i-1] 
                      for i in range(1, len(prices_array))]
            returns_data[coin] = returns
        return returns_data
    except Exception as e:
        print(f"Error calculating returns: {e}")
        return None
    
# ==================== Portfolio Volatility Calculator ====================

def calculate_portfolio_volatility(returns_data: Dict[str, List[float]], weights: Dict[str, float]) -> Optional[Dict]:
    """
    Calculates the volatility (standard deviation) of a portfolio.
    
    Args:
        returns_data (dict): Dictionary of coin returns as lists.
                            Example: {"bitcoin": [0.02, -0.01, 0.03, ...], "ethereum": [0.01, 0.02, -0.02, ...]}
        weights (dict): Portfolio allocation weights (must sum to 1.0).
                       Example: {"bitcoin": 0.6, "ethereum": 0.4}
    
    Returns:
        dict: Portfolio volatility metrics.
              Example: {
                  "portfolio_volatility": 0.45,  # 45% annualized
                  "individual_volatilities": {"bitcoin": 0.50, "ethereum": 0.55},
                  "annualized": True
              }
        None: If calculation fails.
    """
    try:
        # Validate weights sum to 1
        if not np.isclose(sum(weights.values()), 1.0):
            print("Error: Portfolio weights must sum to 1.0")
            return None
        
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(returns_data)
        
        # Calculate covariance matrix (annualized)
        cov_matrix = df.cov() * 252  # 252 trading days per year
        
        # Get weights as array in same order as DataFrame columns
        weight_array = np.array([weights.get(col, 0) for col in df.columns])
        
        # Calculate portfolio variance: w^T * Cov * w
        portfolio_variance = np.dot(weight_array.T, np.dot(cov_matrix, weight_array))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate individual volatilities
        individual_vols = (df.std() * np.sqrt(252)).to_dict()
        
        return {
            "portfolio_volatility": round(portfolio_volatility, 4),
            "portfolio_volatility_pct": round(portfolio_volatility * 100, 2),
            "individual_volatilities": {k: round(v, 4) for k, v in individual_vols.items()},
            "annualized": True
        }
    
    except Exception as e:
        print(f"Error calculating portfolio volatility: {e}")
        return None


# ==================== Value-at-Risk (VaR) Calculator ====================

def calculate_var(returns_data: List[float], confidence_level: float = 0.95, portfolio_value: float = 10000) -> Optional[Dict]:
    """
    Calculates the Value-at-Risk (VaR) for a portfolio or asset.
    
    Args:
        returns_data (list): Historical daily returns as a list of floats.
                            Example: [0.02, -0.01, 0.03, -0.02, 0.01, ...]
        confidence_level (float): Confidence level for VaR (default: 0.95 for 95% VaR).
        portfolio_value (float): Current portfolio value in USD (default: 10000).
    
    Returns:
        dict: VaR metrics.
              Example: {
                  "var_95": 1250.50,  # Maximum loss at 95% confidence in USD
                  "var_95_pct": 12.51,  # As percentage
                  "worst_case_1day": -1800.00,  # Worst historical 1-day loss
                  "confidence_level": 0.95
              }
        None: If calculation fails.
    """
    try:
        returns_array = np.array(returns_data)
        
        # Calculate VaR using historical simulation method
        var_percentile = np.percentile(returns_array, (1 - confidence_level) * 100)
        var_usd = var_percentile * portfolio_value
        var_pct = var_percentile * 100
        
        # Worst case scenario (minimum return in dataset)
        worst_return = np.min(returns_array)
        worst_case_usd = worst_return * portfolio_value
        
        return {
            "var_usd": round(abs(var_usd), 2),
            "var_pct": round(abs(var_pct), 2),
            "worst_case_1day_usd": round(worst_case_usd, 2),
            "worst_case_1day_pct": round(worst_return * 100, 2),
            "confidence_level": confidence_level,
            "portfolio_value": portfolio_value
        }
    
    except Exception as e:
        print(f"Error calculating VaR: {e}")
        return None


# ==================== Portfolio Correlation Matrix ====================

def calculate_correlation_matrix(returns_data: Dict[str, List[float]]) -> Optional[Dict]:
    """
    Calculates the correlation matrix between assets in a portfolio.
    
    Args:
        returns_data (dict): Dictionary of coin returns as lists.
                            Example: {"bitcoin": [0.02, -0.01, ...], "ethereum": [0.01, 0.02, ...]}
    
    Returns:
        dict: Correlation matrix and analysis.
              Example: {
                  "correlation_matrix": {
                      "bitcoin": {"bitcoin": 1.0, "ethereum": 0.85},
                      "ethereum": {"bitcoin": 0.85, "ethereum": 1.0}
                  },
                  "average_correlation": 0.85,
                  "diversification_score": 15  # Lower correlation = better diversification (0-100 scale)
              }
        None: If calculation fails.
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Convert to nested dict format
        corr_dict = {}
        for col in corr_matrix.columns:
            corr_dict[col] = corr_matrix[col].to_dict()
        
        # Calculate average correlation (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_correlation = corr_matrix.where(mask).stack().mean()
        
        # Diversification score (inverse of correlation, scaled 0-100)
        # Lower correlation = higher diversification score
        diversification_score = (1 - avg_correlation) * 100
        
        return {
            "correlation_matrix": {k: {k2: round(v2, 3) for k2, v2 in v.items()} 
                                  for k, v in corr_dict.items()},
            "average_correlation": round(avg_correlation, 3),
            "diversification_score": round(diversification_score, 1),
            "interpretation": "Higher score = better diversification"
        }
    
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        return None


# ==================== Helper Function for Testing ====================

def generate_sample_returns(coins: List[str], days: int = 100, seed: int = 42) -> Dict[str, List[float]]:
    """
    Generates sample return data for testing purposes.
    
    Args:
        coins (list): List of coin names.
        days (int): Number of days of data to generate.
        seed (int): Random seed for reproducibility.
    
    Returns:
        dict: Sample returns data.
    """
    np.random.seed(seed)
    returns_data = {}
    
    for coin in coins:
        # Generate random returns with realistic parameters
        mean_return = 0.001  # 0.1% daily average
        volatility = 0.03    # 3% daily volatility
        returns_data[coin] = list(np.random.normal(mean_return, volatility, days))
    
    return returns_data


# ==================== Test Function ====================

def test_risk_portfolio_tools():
    """
    Tests all Risk & Portfolio Agent functions.
    """
    print("Testing Risk & Portfolio Agent Tools...\n")
    
    # Generate sample data
    coins = ["bitcoin", "ethereum", "solana"]
    returns_data = generate_sample_returns(coins, days=100)
    
    # Test 1: Portfolio Volatility
    print("1. Portfolio Volatility:")
    weights = {"bitcoin": 0.5, "ethereum": 0.3, "solana": 0.2}
    vol_result = calculate_portfolio_volatility(returns_data, weights)
    print(vol_result)
    
    # Test 2: Value-at-Risk
    print("\n2. Value-at-Risk (95% confidence):")
    portfolio_returns = [sum(returns_data[coin][i] * weights[coin] for coin in coins) 
                        for i in range(len(returns_data["bitcoin"]))]
    var_result = calculate_var(portfolio_returns, confidence_level=0.95, portfolio_value=10000)
    print(var_result)
    
    # Test 3: Correlation Matrix
    print("\n3. Correlation Matrix:")
    corr_result = calculate_correlation_matrix(returns_data)
    print(corr_result)


if __name__ == "__main__":
    # Example usage
    test_risk_portfolio_tools()