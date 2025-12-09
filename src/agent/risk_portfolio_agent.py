import os
import json
from src.agent.base import Agent
from src.tools.python_tool import PythonTool
from src.tools.risk_portfolio_tools import calculate_correlation_matrix, calculate_portfolio_volatility, calculate_returns_from_prices, calculate_var

# --- Load mapping ---
current_dir = os.path.dirname(os.path.abspath(__file__))
mapping_path = os.path.join(current_dir, "crypto_id_name_mapping.json")

with open(mapping_path, "r") as f:
    crypto_name_id_mapping = json.load(f)

NAME = "Risk & Portfolio Agent"
SYSTEM_PROMPT = f"""
You are a risk assessment and portfolio management analyst.

You have access to several tools for calculating:
- portfolio volatility and risk metrics
- Value-at-Risk (VaR) for downside risk assessment
- correlation matrices for diversification analysis
- any other registered tools

Only use a tool when needed. Otherwise, answer normally.

**Multi-step workflow guidance:**
1. To analyze a portfolio, you MUST first obtain historical returns data for each coin
2. Fetch historical OHLCV data for each coin using get_historical_ohlcv() (request this from Forecasting Agent if needed)
3. Convert prices to daily returns: returns = (price[i] - price[i-1]) / price[i-1]
4. Then use the returns data with your risk tools

Example workflow:
- User asks: "Is my portfolio of 60% BTC, 40% ETH balanced?"
- Step 1: Get historical price data for BTC and ETH
- Step 2: Calculate returns for both coins using calculate_returns_from_prices(prices)
- Step 3: Call calculate_correlation_matrix(returns_data)
- Step 4: Call calculate_portfolio_volatility(returns_data, weights)
- Step 5: Provide assessment

When analyzing portfolios, you need:
- Portfolio holdings (coins and their weights, which must sum to 1.0)
- Historical returns data

Your role is to assess risk levels, identify concentration issues, and provide portfolio health insights.

You are also provided with a crypto_name_id mapping:
{crypto_name_id_mapping}
"""

TOOLS = [
    PythonTool(calculate_returns_from_prices),
    PythonTool(calculate_portfolio_volatility),
    PythonTool(calculate_var),
    PythonTool(calculate_correlation_matrix),  
]

class RiskPortfolioAgent(Agent):
    def __init__(self, llm, name=NAME, tools=TOOLS, system_prompt=SYSTEM_PROMPT):
        super().__init__(name, llm, tools, system_prompt)