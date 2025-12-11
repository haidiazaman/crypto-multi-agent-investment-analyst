from typing import Dict, List, Optional, Any


def generate_risk_score(volatility: float, var_pct: float, momentum: str, trend_signal: str, correlation_score: Optional[float] = None) -> Optional[Dict]:
    """
    Generates a comprehensive risk score for an investment decision.
    
    Args:
        volatility (float): Portfolio or asset volatility (annualized %).
        var_pct (float): Value-at-Risk percentage.
        momentum (str): Price momentum ("positive", "negative", "neutral").
        trend_signal (str): Trend signal ("bullish", "bearish", "neutral").
        correlation_score (float, optional): Portfolio correlation score (0-100).
    
    Returns:
        dict: Risk assessment with overall score and breakdown.
              Example: {
                  "overall_risk_score": 65,  # 0-100, higher = riskier
                  "risk_level": "medium-high",  # "low", "medium", "medium-high", "high", "very-high"
                  "volatility_risk": 70,
                  "downside_risk": 60,
                  "momentum_risk": 50,
                  "diversification_risk": 75,
                  "recommendation": "Suitable for medium-high risk tolerance investors"
              }
        None: If calculation fails.
    """
    try:
        # Calculate individual risk components (0-100 scale)
        
        # 1. Volatility Risk (higher vol = higher risk)
        volatility_risk = min(volatility * 1.5, 100)  # 67% vol = 100 risk score
        
        # 2. Downside Risk (based on VaR)
        downside_risk = min(var_pct * 5, 100)  # 20% VaR = 100 risk score
        
        # 3. Momentum Risk (negative momentum = higher risk)
        momentum_risk_map = {
            "positive": 30,
            "neutral": 50,
            "negative": 80
        }
        momentum_risk = momentum_risk_map.get(momentum, 50)
        
        # 4. Trend Risk
        trend_risk_map = {
            "bullish": 30,
            "neutral": 50,
            "bearish": 80
        }
        trend_risk = trend_risk_map.get(trend_signal, 50)
        
        # 5. Diversification Risk (if provided)
        if correlation_score is not None:
            # Higher correlation = higher risk (lower diversification)
            diversification_risk = correlation_score
        else:
            diversification_risk = 50  # Neutral if not provided
        
        # Calculate weighted overall risk score
        overall_risk = (
            volatility_risk * 0.30 +
            downside_risk * 0.30 +
            momentum_risk * 0.15 +
            trend_risk * 0.15 +
            diversification_risk * 0.10
        )
        
        # Determine risk level
        if overall_risk < 30:
            risk_level = "low"
            recommendation = "Suitable for conservative investors"
        elif overall_risk < 50:
            risk_level = "medium"
            recommendation = "Suitable for moderate risk tolerance investors"
        elif overall_risk < 70:
            risk_level = "medium-high"
            recommendation = "Suitable for medium-high risk tolerance investors"
        elif overall_risk < 85:
            risk_level = "high"
            recommendation = "Suitable for aggressive investors only"
        else:
            risk_level = "very-high"
            recommendation = "High risk - not suitable for most investors"
        
        return {
            "overall_risk_score": round(overall_risk, 1),
            "risk_level": risk_level,
            "volatility_risk": round(volatility_risk, 1),
            "downside_risk": round(downside_risk, 1),
            "momentum_risk": momentum_risk,
            "trend_risk": trend_risk,
            "diversification_risk": round(diversification_risk, 1),
            "recommendation": recommendation
        }
    
    except Exception as e:
        print(f"Error generating risk score: {e}")
        return None


def generate_investment_recommendation(coin_name: str, current_price: float, risk_score: Dict, technical_signals: Dict, market_sentiment: Optional[str] = None, user_risk_tolerance: str = "medium") -> Optional[Dict]:
    """
    Generates actionable investment recommendations based on all analyses.
    
    Args:
        coin_name (str): Name of the cryptocurrency.
        current_price (float): Current price in USD.
        risk_score (dict): Risk score from generate_risk_score().
        technical_signals (dict): Technical indicators from Forecasting Agent.
        market_sentiment (str, optional): Market sentiment ("bullish", "bearish", "neutral").
        user_risk_tolerance (str): User's risk tolerance ("low", "medium", "high").
    
    Returns:
        dict: Investment recommendation.
              Example: {
                  "action": "BUY",  # "BUY", "HOLD", "SELL", "AVOID"
                  "confidence": "medium",  # "low", "medium", "high"
                  "allocation_suggestion": "5-10% of portfolio",
                  "entry_strategy": "Dollar-cost average over 2-3 weeks",
                  "price_targets": {"short_term": 32000, "medium_term": 35000},
                  "stop_loss": 27000,
                  "reasoning": "Bitcoin shows bullish momentum with RSI at 58...",
                  "risks": ["High volatility", "Regulatory uncertainty"],
                  "match_with_risk_tolerance": True
              }
        None: If generation fails.
    """
    try:
        # Determine if investment matches user's risk tolerance
        risk_tolerance_map = {
            "low": ["low", "medium"],
            "medium": ["low", "medium", "medium-high"],
            "high": ["low", "medium", "medium-high", "high", "very-high"]
        }
        
        risk_match = risk_score["risk_level"] in risk_tolerance_map.get(user_risk_tolerance, ["medium"])
        
        # Determine action based on signals
        bullish_signals = 0
        bearish_signals = 0
        
        # Check technical signals
        if technical_signals.get("trend_signal") == "bullish":
            bullish_signals += 2
        elif technical_signals.get("trend_signal") == "bearish":
            bearish_signals += 2
        
        if technical_signals.get("momentum") == "positive":
            bullish_signals += 1
        elif technical_signals.get("momentum") == "negative":
            bearish_signals += 1
        
        rsi = technical_signals.get("rsi_14")
        if rsi:
            if rsi < 30:  # Oversold
                bullish_signals += 1
            elif rsi > 70:  # Overbought
                bearish_signals += 1
        
        # Check market sentiment
        if market_sentiment == "bullish":
            bullish_signals += 1
        elif market_sentiment == "bearish":
            bearish_signals += 1
        
        # Determine action
        signal_diff = bullish_signals - bearish_signals
        
        if signal_diff >= 3 and risk_match:
            action = "BUY"
            confidence = "high" if signal_diff >= 4 else "medium"
        elif signal_diff >= 1 and risk_match:
            action = "BUY"
            confidence = "medium" if signal_diff >= 2 else "low"
        elif signal_diff <= -3:
            action = "SELL"
            confidence = "high" if signal_diff <= -4 else "medium"
        elif signal_diff <= -1:
            action = "SELL"
            confidence = "medium" if signal_diff <= -2 else "low"
        elif not risk_match:
            action = "AVOID"
            confidence = "high"
        else:
            action = "HOLD"
            confidence = "medium"
        
        # Generate allocation suggestion
        allocation_map = {
            "low": "3-5% of portfolio",
            "medium": "5-10% of portfolio",
            "medium-high": "10-15% of portfolio",
            "high": "15-20% of portfolio",
            "very-high": "Max 5% of portfolio (high risk)"
        }
        allocation = allocation_map.get(risk_score["risk_level"], "5-10% of portfolio")
        
        # Entry strategy
        if action == "BUY":
            if risk_score["overall_risk_score"] > 60:
                entry_strategy = "Dollar-cost average over 3-4 weeks to reduce timing risk"
            else:
                entry_strategy = "Dollar-cost average over 2-3 weeks or lump sum if high conviction"
        else:
            entry_strategy = "N/A"
        
        # Price targets (simple estimation based on current price and trend)
        if technical_signals.get("trend_signal") == "bullish":
            short_term_target = current_price * 1.10
            medium_term_target = current_price * 1.20
        else:
            short_term_target = current_price * 1.05
            medium_term_target = current_price * 1.10
        
        # Stop loss (based on VaR)
        var_pct = risk_score.get("downside_risk", 10) / 5  # Convert risk score back to approx VaR
        stop_loss = current_price * (1 - (var_pct / 100) * 1.5)
        
        # Compile risks
        risks = []
        if risk_score["volatility_risk"] > 60:
            risks.append("High volatility - expect significant price swings")
        if risk_score["downside_risk"] > 60:
            risks.append("Significant downside risk")
        if not risk_match:
            risks.append(f"Risk level ({risk_score['risk_level']}) may not match your tolerance ({user_risk_tolerance})")
        
        # Reasoning
        reasoning = f"{coin_name} "
        if action == "BUY":
            reasoning += f"shows {bullish_signals} bullish signals vs {bearish_signals} bearish signals. "
        elif action == "SELL":
            reasoning += f"shows {bearish_signals} bearish signals vs {bullish_signals} bullish signals. "
        else:
            reasoning += f"shows mixed signals ({bullish_signals} bullish, {bearish_signals} bearish). "
        
        reasoning += f"Risk level: {risk_score['risk_level']}. "
        if rsi:
            reasoning += f"RSI at {rsi}. "
        
        return {
            "action": action,
            "confidence": confidence,
            "allocation_suggestion": allocation if action in ["BUY", "HOLD"] else "N/A",
            "entry_strategy": entry_strategy,
            "price_targets": {
                "short_term": round(short_term_target, 2),
                "medium_term": round(medium_term_target, 2)
            } if action == "BUY" else None,
            "stop_loss": round(stop_loss, 2) if action == "BUY" else None,
            "reasoning": reasoning,
            "risks": risks,
            "match_with_risk_tolerance": risk_match
        }
    
    except Exception as e:
        print(f"Error generating investment recommendation: {e}")
        return None


def test_synthesis_recommendation_tools():
    """
    Tests all Synthesis & Recommendation Agent functions.
    """
    print("Testing Synthesis & Recommendation Agent Tools...\n")
    
    # Sample data for testing
    data_points = [
        {
            "source": "Market Data",
            "data": {"price": 30000, "24h_change": 5.2, "volume": 25000000000}
        },
        {
            "source": "Technical Analysis",
            "data": {"rsi_14": 58, "trend_signal": "bullish", "momentum": "positive"}
        },
        {
            "source": "Risk Metrics",
            "data": {"volatility": 45, "var_95": 12.5}
        }
    ]
    
    # Test 1: Generate Risk Score
    print("1. Generating Risk Score:")
    risk_score = generate_risk_score(
        volatility=45,
        var_pct=12.5,
        momentum="positive",
        trend_signal="bullish",
        correlation_score=65
    )
    print(risk_score)
    
    # Test 2: Generate Investment Recommendation
    print("\n2. Generating Investment Recommendation:")
    technical_signals = {
        "rsi_14": 58,
        "trend_signal": "bullish",
        "momentum": "positive"
    }
    recommendation = generate_investment_recommendation(
        coin_name="Bitcoin",
        current_price=30000,
        risk_score=risk_score,
        technical_signals=technical_signals,
        market_sentiment="bullish",
        user_risk_tolerance="medium"
    )
    print(recommendation)
    
    # # Test 3: Summarize Findings (only if OpenAI API key is set)
    # if OPENAI_API_KEY != "your_openai_api_key_here":
    #     print("\n3. Summarizing Findings:")
    #     summary = summarize_findings(data_points)
    #     print(summary)
    # else:
    #     print("\n3. Skipping OpenAI summarization test (API key not set)")


if __name__ == "__main__":
    # Example usage
    test_synthesis_recommendation_tools()