import requests
import numpy as np
from typing import Optional, Dict, List

def get_historical_close_prices_and_volumes(coin_id: str = "bitcoin", vs_currency: str = "usd", days: int = 30) -> Optional[Dict]:
    """
    Fetches historical Close prices and Volume data for a cryptocurrency. 
    Only daily interval prices can be fetched.
    
    Args:
        coin_id (str): CoinGecko coin ID (e.g., "bitcoin", "ethereum").
        vs_currency (str): Currency to price against (default: "usd").
        days (int): Number of days of historical data (default: 30).
    
    Returns:
        dict: Historical price and volume data.
              Example for bitcoin 30 days: {
                  "coin_id": "bitcoin",
                  "days": 30
                  "prices": [...],
                  "volumes": [...],
              }
        None: If the request fails.
    """
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": days,
            "interval": "daily"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        prices = [price[1] for price in data["prices"]]
        volumes = [volume[1] for volume in data["total_volumes"]]

        result = {}
        result["coin_id"] = coin_id
        result["days"] = days
        result['prices'] = prices
        result['volumes'] = volumes
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical OHLCV data: {e}")
        return None

def calculate_technical_indicators(prices: List[float]) -> Optional[Dict]:
    """
    Calculates common technical indicators (RSI, SMA, EMA) from price data.
    
    Args:
        prices (list): List of historical closing prices (chronological order, oldest first).
                      Example: [30000, 31000, 29500, 32000, ...]
    
    Returns:
        dict: Technical indicators.
              Example: {
                  "rsi_14": 58.5,  # 14-period RSI
                  "sma_20": 30500,  # 20-period Simple Moving Average
                  "sma_50": 29800,  # 50-period Simple Moving Average
                  "ema_12": 31200,  # 12-period Exponential Moving Average
                  "current_price": 32000,
                  "price_vs_sma20": "above",  # "above" or "below"
                  "trend_signal": "bullish"  # "bullish", "bearish", or "neutral"
              }
        None: If calculation fails.
    """
    try:
        if len(prices) < 50:
            print("Warning: Need at least 50 data points for accurate indicators")
        
        prices_array = np.array(prices)
        current_price = prices_array[-1]
        
        # Calculate RSI (14-period)
        def calculate_rsi(prices, period=14):
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # First average
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            # EMA for subsequent values
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi_14 = calculate_rsi(prices_array, 14) if len(prices) >= 15 else None
        
        # Calculate SMAs
        sma_20 = np.mean(prices_array[-20:]) if len(prices) >= 20 else None
        sma_50 = np.mean(prices_array[-50:]) if len(prices) >= 50 else None
        
        # Calculate EMA (12-period)
        def calculate_ema(prices, period=12):
            ema = [prices[0]]
            multiplier = 2 / (period + 1)
            for price in prices[1:]:
                ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
            return ema[-1]
        
        ema_12 = calculate_ema(prices_array, 12) if len(prices) >= 12 else None
        
        # Determine trend signal
        trend_signal = "neutral"
        if sma_20 and sma_50:
            if current_price > sma_20 > sma_50:
                trend_signal = "bullish"
            elif current_price < sma_20 < sma_50:
                trend_signal = "bearish"
        
        price_vs_sma20 = "above" if sma_20 and current_price > sma_20 else "below"
        
        return {
            "rsi_14": round(rsi_14.item(), 2) if rsi_14 else None,
            "sma_20": round(sma_20.item(), 2) if sma_20 else None,
            "sma_50": round(sma_50.item(), 2) if sma_50 else None,
            "ema_12": round(ema_12.item(), 2) if ema_12 else None,
            "current_price": round(current_price.item(), 2),
            "price_vs_sma20": price_vs_sma20 if sma_20 else None,
            "trend_signal": trend_signal
        }
    
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return None

def analyze_price_volume_trend(prices: List[float], volumes: Optional[List[float]] = None) -> Optional[Dict]:
    """
    Analyzes price trends and momentum over different time periods. Volume can also be passed.
    
    Args:
        prices (list): List of historical closing prices (chronological order).
        volumes (list, optional): List of trading volumes corresponding to prices.
    
    Returns:
        dict: Trend analysis metrics.
              Example: {
                  "7d_return": 5.2,  # 7-day return percentage
                  "30d_return": 15.8,
                  "volatility_30d": 45.5,  # Annualized volatility percentage
                  "momentum": "positive",  # "positive", "negative", or "neutral"
                  "strength": "strong",  # "strong", "moderate", "weak"
                  "volume_trend": "increasing"  # "increasing", "decreasing", "stable"
              }
        None: If calculation fails.
    """
    try:
        prices_array = np.array(prices)
        current_price = prices_array[-1]
        
        # Calculate returns for different periods
        def calculate_return(start_idx):
            if len(prices_array) > abs(start_idx):
                start_price = prices_array[start_idx]
                return ((current_price - start_price) / start_price) * 100
            return None
        
        return_7d = calculate_return(-7)
        return_30d = calculate_return(-30)
        
        # Calculate volatility (annualized)
        if len(prices_array) >= 30:
            returns = np.diff(prices_array) / prices_array[:-1]
            volatility_30d = np.std(returns[-30:]) * np.sqrt(252) * 100
        else:
            volatility_30d = None
        
        # Determine momentum
        momentum = "neutral"
        strength = "weak"
        
        if return_7d is not None:
            if return_7d > 5:
                momentum = "positive"
                strength = "strong" if return_7d > 10 else "moderate"
            elif return_7d < -5:
                momentum = "negative"
                strength = "strong" if return_7d < -10 else "moderate"
        
        # Volume trend analysis
        volume_trend = None
        if volumes and len(volumes) >= 14:
            recent_vol_avg = np.mean(volumes[-7:])
            previous_vol_avg = np.mean(volumes[-14:-7])
            vol_change = ((recent_vol_avg - previous_vol_avg) / previous_vol_avg) * 100
            
            if vol_change > 20:
                volume_trend = "increasing"
            elif vol_change < -20:
                volume_trend = "decreasing"
            else:
                volume_trend = "stable"
        
        return {
            "7d_return": round(return_7d.item(), 2) if return_7d else None,
            "30d_return": round(return_30d.item(), 2) if return_30d else None,
            "volatility_30d": round(volatility_30d.item(), 2) if volatility_30d else None,
            "momentum": momentum,
            "strength": strength,
            "volume_trend": volume_trend,
            "current_price": round(current_price.item(), 2)
        }
    
    except Exception as e:
        print(f"Error analyzing price trend: {e}")
        return None


# ==================== Test Function ====================

def test_forecasting_analysis_tools():
    """
    Tests all Forecasting & Analysis Agent functions.
    """
    print("Testing Forecasting & Analysis Agent Tools...\n")
    
    # Test 1: Get Historical OHLCV
    print("1. Fetching Historical OHLCV for Bitcoin (30 days):")
    data = get_historical_close_prices_and_volumes("bitcoin", "usd", 30)
    if data:
        prices, volumes = data['prices'], data['volumes']
        print(f"Data points: {len(prices)}")
        print(f"Price range: ${min(prices):,.2f} - ${max(prices):,.2f}")
    
    # Test 2: Technical Indicators
    if prices:
        print("\n2. Calculating Technical Indicators:")
        indicators = calculate_technical_indicators(prices)
        print(indicators)
    
    # Test 3: Price Trend Analysis
    if prices and volumes:
        print("\n3. Analyzing Price Trend:")
        trend = analyze_price_volume_trend(prices, volumes)
        print(trend)


if __name__ == "__main__":
    test_forecasting_analysis_tools()