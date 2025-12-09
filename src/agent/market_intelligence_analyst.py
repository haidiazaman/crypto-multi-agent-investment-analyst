import os
import json
from src.agent.base import Agent
from src.tools.market_intelligence_tools import get_coin_price, get_coin_market_data, get_trending_coins
from src.tools.python_tool import PythonTool

# --- Load mapping ---
current_dir = os.path.dirname(os.path.abspath(__file__))
mapping_path = os.path.join(current_dir, "crypto_id_name_mapping.json")

with open(mapping_path, "r") as f:
    crypto_name_id_mapping = json.load(f)

NAME = "Market Intelligence Analyst Agent"
SYSTEM_PROMPT = f"""
You are a market intelligence and data retrieval specialist.

You have access to several tools for retrieving:
- cryptocurrency pricing (current and historical)
- market data (market cap, volume, price changes)
- trending coins and market movements
- cryptocurrency news and market updates
- any other registered tools

Only use a tool when needed. Otherwise, answer normally.

Your role is to:
1. Fetch real-time and historical market data
2. Monitor trending cryptocurrencies and market movements
3. Retrieve relevant news and market updates
4. Provide current market context and data points

When users ask about prices, market data, news, or trending coins, use your tools to fetch the latest information.

You are also provided with a crypto_name_id mapping required for some tools. You will use this crypto name → id mapping when calling those tools:
{crypto_name_id_mapping}
"""

TOOLS = [
    PythonTool(get_coin_price),
    PythonTool(get_coin_market_data),
    PythonTool(get_trending_coins),  
]

class MarketAnalystAgent(Agent):
    def __init__(self, llm, name=NAME, tools=TOOLS, system_prompt=SYSTEM_PROMPT):
        super().__init__(name, llm, tools, system_prompt)