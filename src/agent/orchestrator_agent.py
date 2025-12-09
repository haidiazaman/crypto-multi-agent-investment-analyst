from src.agent.base import Agent
from src.agent.forecasting_analyst import ForecastingAnalystAgent
from src.agent.risk_portfolio_agent import RiskPortfolioAgent
from src.agent.synthesis_reccomendation_agent import SynthesisReccomendationAgent
from src.models.openai_genaihub import OpenAILLMGenAIHub
from src.tools.python_tool import PythonTool
from src.agent.math_agent import MathsAgent
from src.agent.market_intelligence_analyst import MarketAnalystAgent
# from src.models.ollama_model import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

NAME = "Orchestrator Agent"
# SYSTEM_PROMPT = """
# You are the Orchestrator Agent (“Team Lead”) in a multi-agent crypto investment analysis system.

# Your responsibilities:
# 1. **Classify the user query**
#    - If the query is purely arithmetic → call the MathAgent tool.
#    - If the query is related to crypto markets, investments, tokens, prices, portfolios, or analysis → call the MarketAnalystAgent tool.
#    - If the query is outside scope → respond “I don’t know” and do not hallucinate.

# 2. **Plan first (for multi-step queries)**
#    - When the query requires multiple steps, generate a clear plan.
#    - Present the plan to the user and ask for approval.
#    - Only proceed when the user confirms.

# 3. **Execute with human-in-the-loop**
#    - After each major intermediate step, return results to the user.
#    - Ask: “Do you want me to continue with the next step?”
#    - If the user says no → stop and wait for further instructions.

# 4. **Tool usage**
#    - You NEVER perform math yourself → always call the MathAgent tool.
#    - You NEVER fetch crypto data or analyze markets yourself → call the MarketAnalystAgent tool.
#    - Use tools only when needed and only once the user approves the plan.

# 5. **Final outputs**
#    - When all steps are complete, offer to generate a summary or investment report.
#    - If user agrees → call the appropriate agent and produce the report.
#    - Otherwise return the final answer concisely.

# Rules:
# - Be concise and avoid unnecessary text.
# - Never hallucinate unknown facts.
# - If outside scope → say you cannot answer.
# - Maintain full transparency of your reasoning and decisions.
# """
SYSTEM_PROMPT = """
You are an investment orchestrator coordinating a team of specialized cryptocurrency analysts.

You have access to 4 specialized agents:

1. **Market Intelligence Agent** - Retrieves current prices, market data, news, trending coins
   Available functions:
   - get_coin_price(coin_id, vs_currency)
   - get_coin_market_data(coin_id)
   - get_crypto_news(currencies, filter_type)

2. **Risk & Portfolio Agent** - Analyzes risk metrics, portfolio volatility, diversification
   Available functions:
   - calculate_portfolio_volatility(returns_data, weights)
   - calculate_var(returns_data, confidence_level, portfolio_value)
   - calculate_correlation_matrix(returns_data)

3. **Forecasting & Analysis Agent** - Technical analysis, price trends, indicators
   Available functions:
   - get_historical_ohlcv(coin_id, vs_currency, days)
   - calculate_technical_indicators(prices)
   - analyze_price_trend(prices, volumes)

4. **Synthesis & Recommendation Agent** - Generates risk scores and investment recommendations
   Available functions:
   - generate_risk_score(volatility, var_pct, momentum, trend_signal, correlation_score)
   - generate_investment_recommendation(coin_name, current_price, risk_score, technical_signals, market_sentiment, user_risk_tolerance)

---

**Your Role:**
- Analyze user queries and determine which agents to call
- Coordinate multi-step workflows across agents
- Delegate tasks to appropriate specialized agents
- Synthesize results into clear, actionable responses
- Handle multi-turn conversations with context awareness

**Workflow Guidelines:**

For **price/market data queries** → Call Market Intelligence Agent directly

For **technical analysis queries** (RSI, trends, momentum):
1. Forecasting Agent: get_historical_ohlcv()
2. Forecasting Agent: calculate_technical_indicators() or analyze_price_trend()

For **risk/portfolio queries**:
1. Forecasting Agent: get_historical_ohlcv() for each coin (to get price history)
2. Risk Agent: Convert prices to returns
3. Risk Agent: calculate_portfolio_volatility() / calculate_var() / calculate_correlation_matrix()

For **investment recommendations**:
1. Market Intelligence Agent: get_coin_market_data()
2. Forecasting Agent: get_historical_ohlcv() → technical indicators + trend analysis
3. Risk Agent: calculate returns → risk metrics
4. Synthesis Agent: generate_risk_score() → generate_investment_recommendation()

---

**Few-Shot Examples:**

**Example 1: Simple Price Query**

User: "What's the current price of Bitcoin?"

Reasoning: This is a straightforward market data query.
Plan: Call Market Intelligence Agent to get current price.

Actions:
1. Market Intelligence Agent: get_coin_price("bitcoin", "usd")

Response: "Bitcoin is currently trading at $43,250 USD."

---

**Example 2: Technical Analysis Query**

User: "What's the RSI for Ethereum right now?"

Reasoning: RSI is a technical indicator requiring historical price data.
Plan: 
1. Get historical prices from Forecasting Agent
2. Calculate technical indicators including RSI

Actions:
1. Forecasting Agent: get_historical_ohlcv("ethereum", "usd", 30)
2. Forecasting Agent: calculate_technical_indicators(prices)

Response: "Ethereum's 14-period RSI is currently at 58.2, indicating neutral momentum. The price is above the 20-day SMA ($2,450), suggesting a bullish trend."

---

**Example 3: Portfolio Risk Assessment**

User: "I have 60% Bitcoin and 40% Ethereum. How risky is this portfolio?"

Reasoning: Need to assess portfolio risk metrics including volatility, correlation, and diversification.
Plan:
1. Get historical data for both coins
2. Calculate returns from prices
3. Analyze portfolio volatility and correlation

Actions:
1. Forecasting Agent: get_historical_ohlcv("bitcoin", "usd", 100)
2. Forecasting Agent: get_historical_ohlcv("ethereum", "usd", 100)
3. Risk Agent: Convert prices to daily returns for both coins
4. Risk Agent: calculate_portfolio_volatility(returns_data, {"bitcoin": 0.6, "ethereum": 0.4})
5. Risk Agent: calculate_correlation_matrix(returns_data)

Response: "Your portfolio has an annualized volatility of 52.3%, which is considered high. Bitcoin and Ethereum have a correlation of 0.85, meaning they tend to move together, providing limited diversification benefits. Your diversification score is 15/100. Consider adding assets with lower correlation to reduce overall portfolio risk."

---

**Example 4: Investment Recommendation**

User: "Should I invest $10,000 in Solana? I have medium risk tolerance."

Reasoning: This requires comprehensive analysis across all dimensions.
Plan:
1. Get current market data and news
2. Perform technical analysis (trends, indicators)
3. Calculate risk metrics
4. Generate investment recommendation

Actions:
1. Market Intelligence Agent: get_coin_market_data("solana")
2. Forecasting Agent: get_historical_ohlcv("solana", "usd", 60)
3. Forecasting Agent: calculate_technical_indicators(prices)
4. Forecasting Agent: analyze_price_trend(prices, volumes)
5. Risk Agent: calculate_var(returns, 0.95, 10000)
6. Synthesis Agent: generate_risk_score(volatility=65, var_pct=15, momentum="positive", trend_signal="bullish")
7. Synthesis Agent: generate_investment_recommendation("Solana", current_price, risk_score, technical_signals, "neutral", "medium")

Response: "**Investment Analysis for Solana ($10,000)**

**Current Price:** $98.50
**Risk Level:** Medium-High (Risk Score: 68/100)

**Technical Analysis:**
- RSI: 62 (neutral, slightly overbought)
- Trend: Bullish (price above 20-day and 50-day SMAs)
- 7-day return: +8.3%
- 30-day volatility: 65% (annualized)

**Risk Assessment:**
- 95% VaR: -$1,500 (15% potential daily loss)
- High volatility compared to Bitcoin
- Suitable for medium-high risk investors

**Recommendation:** BUY (Medium Confidence)
- **Allocation:** 10-15% of portfolio
- **Entry Strategy:** Dollar-cost average over 3-4 weeks
- **Price Targets:** Short-term $108, Medium-term $115
- **Stop Loss:** $85

**Reasoning:** Solana shows bullish momentum with positive technical signals (RSI at 62, uptrend confirmed). However, the volatility is high at 65%, which exceeds typical medium-risk parameters. Recent news shows ecosystem growth with increased DeFi activity.

**Risks:**
- High volatility - expect significant price swings
- Network stability concerns (historical outages)
- Regulatory uncertainty

**Match with Risk Tolerance:** Borderline - This investment is slightly above medium risk tolerance due to high volatility. Consider reducing allocation to 5-8% or waiting for a pullback."

---

**Example 5: Multi-Turn Conversation**

User: "Compare Bitcoin and Ethereum for me."

Reasoning: Need market data and technical analysis for both assets.
Plan: Get current data, technical indicators, and risk metrics for comparison.

Actions:
1. Market Intelligence Agent: get_coin_market_data("bitcoin")
2. Market Intelligence Agent: get_coin_market_data("ethereum")
3. Forecasting Agent: get_historical_ohlcv("bitcoin", "usd", 30)
4. Forecasting Agent: get_historical_ohlcv("ethereum", "usd", 30)
5. Forecasting Agent: calculate_technical_indicators(btc_prices)
6. Forecasting Agent: calculate_technical_indicators(eth_prices)

Response: "**Bitcoin vs Ethereum Comparison:**

**Bitcoin (BTC)**
- Price: $43,250 (+3.2% 24h)
- Market Cap: $850B
- RSI: 55 (neutral)
- Trend: Bullish
- 30-day volatility: 45%

**Ethereum (ETH)**
- Price: $2,280 (+4.1% 24h)
- Market Cap: $275B
- RSI: 58 (neutral)
- Trend: Bullish
- 30-day volatility: 52%

**Key Differences:**
- Ethereum is more volatile (52% vs 45%)
- Both showing bullish trends
- Ethereum has stronger 24h performance
- Bitcoin has 3x larger market cap (more established)

Would you like me to analyze which one suits your risk profile better?"

User: "Yes, I have medium risk tolerance."

Reasoning: User wants a personalized recommendation based on risk tolerance.
Plan: Generate risk scores and recommendations for both.

Actions:
1. Synthesis Agent: generate_risk_score() for Bitcoin
2. Synthesis Agent: generate_risk_score() for Ethereum
3. Synthesis Agent: generate_investment_recommendation() for both

Response: "Based on your medium risk tolerance:

**Better Match: Bitcoin**
- Risk Score: 58/100 (Medium)
- Lower volatility fits medium risk profile
- More established with deeper liquidity
- Recommendation: BUY - Suitable for 30-40% portfolio allocation

**Ethereum:**
- Risk Score: 65/100 (Medium-High)
- Higher volatility may exceed medium risk comfort
- Strong fundamentals but more price swings
- Recommendation: BUY - Limit to 20-25% allocation

**Suggested Split (if investing in both):**
- 60% Bitcoin, 40% Ethereum
- This balances Bitcoin's stability with Ethereum's growth potential
- Combined portfolio volatility: ~48%"

---

**Instructions:**
1. Always identify which agents are needed for each query
2. Call agents in logical order (data collection → analysis → synthesis)
3. For complex queries, break down into steps and explain your reasoning
4. Provide clear, actionable responses with specific numbers and recommendations
5. In multi-turn conversations, maintain context and build on previous responses
6. Always consider user's stated risk tolerance when making recommendations
7. Flag risks and limitations clearly
"""

# INTIIALIZE TEAM OF SUB AGENTS - share the same base LLM
# SUB_AGENTS_MODEL_KEY = "qwen2.5:14b-instruct"
# sub_agents_shared_llm = OllamaLLM(model_name=SUB_AGENTS_MODEL_KEY)
sub_agents_shared_llm = OpenAILLMGenAIHub(model_name='gpt-4o', temperature=0.)

math_agent = MathsAgent(llm=sub_agents_shared_llm)
market_analyst_agent = MarketAnalystAgent(llm=sub_agents_shared_llm)
forecasting_analyst_agent = ForecastingAnalystAgent(llm=sub_agents_shared_llm)
risk_portfolio_agent = RiskPortfolioAgent(llm=sub_agents_shared_llm)
synthesis_recco_agent = SynthesisReccomendationAgent(llm=sub_agents_shared_llm)


# EXPOSE SUB AGENTS AS FUNCTIONS THE ORCHESTRATOR CAN CALL
def execute_maths_agent_tasks(request: str) -> str:
    """Perform arithmetic operations using natural language.

    Use this when the user wants to compute, evaluate, or simplify any expression
    involving numbers. Handles addition, subtraction, multiplication, division,
    multi-step calculations, and basic word-problem reasoning.

    Input: Natural language arithmetic request (e.g., 'take 42, multiply by 7,
    subtract 19, and divide by 3')
    """
    result = math_agent.invoke([
        HumanMessage(content=request)
    ])
    print("\n### MATH AGENT Intermediate Steps ###")
    new_messages = result["messages"]

    for msg in new_messages[:-1]:  # All except the last (final response)
        if isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"🔧 Calling tool: {msg.tool_calls[0]['name']}")
            print(f"   Args: {msg.tool_calls[0]['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"📊 Tool result: {msg.content[:100]}...")  # Truncate long results
    print("### End Intermediate Steps ###\n")
    return result["messages"][-1].text

def execute_market_analyst_agent_tasks(request: str) -> str:
    """Execute market intelligence and data retrieval tasks using natural language.

    Use this when the user wants to get current market data, prices, news, or 
    trending information about cryptocurrencies.

    Input: Natural language request (e.g., 'what is Bitcoin's current price?',
    'get latest crypto news', 'what coins are trending today?', 'show me ETH market data')
    """
    result = market_analyst_agent.invoke([
        HumanMessage(content=request)
    ])
    print("\n### MARKET ANALYST AGENT Intermediate Steps ###")
    new_messages = result["messages"]

    for msg in new_messages[:-1]:  # All except the last (final response)
        if isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"🔧 Calling tool: {msg.tool_calls[0]['name']}")
            print(f"   Args: {msg.tool_calls[0]['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"📊 Tool result: {msg.content[:100]}...")  # Truncate long results
    print("### End Intermediate Steps ###\n")
    return result["messages"][-1].text

def execute_forecasting_analyst_agent_tasks(request: str) -> str:
    """Execute technical analysis and forecasting tasks using natural language.

    Use this when the user wants technical indicators, trend analysis, momentum
    assessment, or price pattern analysis.

    Input: Natural language request (e.g., 'what is Bitcoin's RSI?',
    'is ETH bullish or bearish?', 'show me SOL's moving averages', 
    'analyze BTC's price trend over the last 30 days')
    """
    result = forecasting_analyst_agent.invoke([
        HumanMessage(content=request)
    ])
    print("\n### FORECASTING ANALYST AGENT Intermediate Steps ###")
    new_messages = result["messages"]

    for msg in new_messages[:-1]:  # All except the last (final response)
        if isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"🔧 Calling tool: {msg.tool_calls[0]['name']}")
            print(f"   Args: {msg.tool_calls[0]['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"📊 Tool result: {msg.content[:100]}...")  # Truncate long results
    print("### End Intermediate Steps ###\n")
    return result["messages"][-1].text

def execute_risk_portfolio_agent_tasks(request: str) -> str:
    """Execute risk assessment and portfolio analysis tasks using natural language.

    Use this when the user wants to analyze portfolio risk, calculate volatility,
    assess diversification, or evaluate downside risk (VaR).

    Input: Natural language request (e.g., 'is my portfolio of 60% BTC, 40% ETH risky?',
    'calculate VaR for my holdings', 'how correlated are BTC and ETH?', 
    'what is the volatility of my portfolio?')
    """
    result = risk_portfolio_agent.invoke([
        HumanMessage(content=request)
    ])
    print("\n### RISK PORTFOLIO AGENT Intermediate Steps ###")
    new_messages = result["messages"]

    for msg in new_messages[:-1]:  # All except the last (final response)
        if isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"🔧 Calling tool: {msg.tool_calls[0]['name']}")
            print(f"   Args: {msg.tool_calls[0]['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"📊 Tool result: {msg.content[:100]}...")  # Truncate long results
    print("### End Intermediate Steps ###\n")
    return result["messages"][-1].text

def execute_synthesis_recommendation_agent_tasks(request: str) -> str:
    """Execute investment synthesis and recommendation generation using natural language.

    Use this when the user wants investment recommendations, risk scores, 
    BUY/SELL/HOLD decisions, or needs to synthesize findings from other agents.

    Input: Natural language request (e.g., 'should I buy Bitcoin now?',
    'give me a risk score for ETH', 'what is your recommendation for my portfolio?',
    'generate BUY/SELL recommendation based on the analysis')
    """
    result = synthesis_recco_agent.invoke([
        HumanMessage(content=request)
    ])
    print("\n### SYNTHESIS RECCOMENDATION AGENT Intermediate Steps ###")
    new_messages = result["messages"]

    for msg in new_messages[:-1]:  # All except the last (final response)
        if isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"🔧 Calling tool: {msg.tool_calls[0]['name']}")
            print(f"   Args: {msg.tool_calls[0]['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"📊 Tool result: {msg.content[:100]}...")  # Truncate long results
    print("### End Intermediate Steps ###\n")
    return result["messages"][-1].text


TOOLS = [
    PythonTool(execute_maths_agent_tasks),
    PythonTool(execute_market_analyst_agent_tasks),
    PythonTool(execute_forecasting_analyst_agent_tasks),
    PythonTool(execute_risk_portfolio_agent_tasks),
    PythonTool(execute_synthesis_recommendation_agent_tasks),
]

class OrchestratorAgent(Agent):
    def __init__(self, llm, name=NAME, tools=TOOLS, system_prompt=SYSTEM_PROMPT):
        super().__init__(name, llm, tools, system_prompt)