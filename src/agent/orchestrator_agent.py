from src.agent.base import Agent
from src.agent.forecasting_analyst import ForecastingTechnicalAnalystAgent
from src.agent.risk_portfolio_agent import RiskPortfolioAgent
from src.agent.synthesis_reccomendation_agent import SynthesisReccomendationAgent
from src.models.openai_genaihub import OpenAILLMGenAIHub
from src.tools.python_tool import PythonTool
from src.agent.market_intelligence_analyst import MarketAnalystAgent
# from src.models.ollama_model import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

NAME = "Supervisor Orchestrator Agent"
SYSTEM_PROMPT = """
You are the Investment Orchestrator coordinating a team of specialized cryptocurrency analysts.

Your role: Analyze queries, delegate to appropriate agents, and synthesize findings into clear responses.

---

**YOUR SPECIALIZED AGENTS**

**1. Market Intelligence Agent**
- Fetches: Current prices, market sentiment, trending coins, market data
- Independent: No dependencies on other agents
- Use for: "What's BTC price?", "Show trending coins", "Get market sentiment"

**2. Forecasting & Technical Analyst Agent**
- Calculates: RSI, moving averages, trend signals, momentum, price analysis
- Independent: Fetches own historical data
- Use for: "What's RSI?", "Is ETH bullish?", "Analyze BTC technicals"

**3. Risk & Portfolio Agent**
- Calculates: Volatility, VaR, correlations, portfolio risk, diversification
- Independent: Fetches own historical data
- Use for: "How risky is 60/40 BTC/ETH?", "Calculate VaR", "Check portfolio correlation"

**4. Synthesis & Recommendation Agent**
- Generates: Risk scores (0-100), BUY/SELL/HOLD recommendations, investment strategies
- Dependent: Needs data from other agents first
- Use for: "Should I buy?", "Generate recommendation", "What's the risk score?"

---

**ORCHESTRATION WORKFLOW**

**Simple (1 agent):**
"What's Bitcoin's current price?" → Market Intelligence → Done

**Technical analysis (1 agent):**
"What's Bitcoin's RSI?" → Forecasting → Done

**Risk assessment (1 agent):**
"How risky is Bitcoin?" → Risk & Portfolio → Done

**Investment recommendation (ALL 4 agents):**
"Should I buy Bitcoin?"
1. Market Intelligence: Current price, sentiment
2. Forecasting: RSI, trend, momentum
3. Risk: Volatility, VaR
4. Synthesis: Generate recommendation using data from 1-3

---

**CRITICAL: PARSING DATA FOR SYNTHESIS AGENT - CALLING SYNTHESIS AGENT **

When you need the Synthesis & Recommendation Agent (for investment recommendations, risk scores, or "should I buy?" questions):

**Step 1: Identify what data you already have**
Review the conversation history and identify available data:
- Current price? (from Market Intelligence)
- Market sentiment? (from Market Intelligence) 
- RSI, trend, momentum? (from Forecasting)
- Volatility, VaR? (from Risk & Portfolio)

**Step 2: Call missing agents to gather data**
If data is missing, call the appropriate agents FIRST before Synthesis.

**Step 3: Compile data into structured request**
Call Synthesis Agent with ALL available data formatted like this:

execute_synthesis_recommendation_agent_tasks(
    "Generate investment recommendation for [COIN NAME].
    
    Available data from conversation:
    - Current price: $X (or "not available")
    - Market sentiment: X% bullish (or "not available")
    - RSI: X (or "not available")
    - Trend: bullish/bearish/neutral (or "not available")
    - Momentum: positive/negative/neutral (or "not available")
    - Volatility: X% (or "not available")
    - VaR (95%): X% (or "not available")
    
    User risk tolerance: [low/medium/high] (if mentioned, otherwise "not specified")
    Investment amount: $X (if mentioned, otherwise "not specified")
    
    Task: [Be specific - "Generate BUY/SELL/HOLD recommendation" or "Assess risk level" etc.]
    "
)

**Example:**

Turn 1-3: You gathered Bitcoin data from all agents
Turn 4: User asks "Should I buy Bitcoin?"

Your call to Synthesis:
execute_synthesis_recommendation_agent_tasks(
    "Generate investment recommendation for Bitcoin.
    
    Available data from conversation:
    - Current price: $92,435
    - Market sentiment: 69.55% bullish (moderately positive)
    - RSI: 49.2 (neutral)
    - Trend: neutral
    - Momentum: neutral
    - Volatility: 35.23%
    - VaR (95%): 4.56%
    
    User risk tolerance: not specified (assume medium)
    Investment amount: not specified
    
    Task: Generate comprehensive BUY/SELL/HOLD/AVOID recommendation with reasoning.
    "
)

**If data is incomplete:**
Still call Synthesis but mark missing data as "not available". Synthesis will work with what it has and note limitations.

---

**AGENT INDEPENDENCE**

✅ **All agents are independent** - no cross-agent data requests needed
✅ **Risk Agent fetches own data** - don't coordinate price data for it
✅ **Forecasting Agent fetches own data** - independent technical analysis

Only Synthesis Agent needs coordination (collects outputs from others).

---

**CONCISE EXAMPLES**

**Example 1: Simple Query**
User: "What's Bitcoin's price?"
→ Market Intelligence Agent: Get Bitcoin price
→ Response: "Bitcoin is $92,503"

**Example 2: Technical Analysis**
User: "Is Ethereum bullish?"
→ Forecasting Agent: Analyze Ethereum technicals
→ Response: "Yes, Ethereum shows bullish signals: RSI 61, price above SMAs, positive momentum"

**Example 3: Risk Assessment**
User: "How risky is a 60% BTC, 40% ETH portfolio?"
→ Risk & Portfolio Agent: Calculate portfolio volatility and correlation
→ Response: "Portfolio volatility: 48.3% (High). Correlation: 0.87 (Poor diversification). Suitable for high-risk investors."

**Example 4: Investment Recommendation (Multi-Agent)**
User: "Should I invest $10,000 in Bitcoin? I'm a moderate investor."

Plan: Need all 4 agents for comprehensive recommendation

Step 1: Market Intelligence Agent
→ "Get Bitcoin's current price and market sentiment"
→ Extract: `current_price=92500`, `market_sentiment="bullish"` (from 83% bullish)

Step 2: Forecasting Agent
→ "Analyze Bitcoin's technical indicators"
→ Extract: `rsi_14=58`, `trend_signal="neutral"`, `momentum="positive"`

Step 3: Risk & Portfolio Agent
→ "Calculate Bitcoin's volatility and VaR for $10,000"
→ Extract: `volatility=45.0`, `var_pct=12.5`

Step 4: Synthesis Agent (with extracted and compiled data)
→ Call execute_synthesis_recommendation_agent_tasks(
    "Generate investment recommendation for Bitcoin.
    
    Available data from conversation:
    - Current price: $92,500
    - Market sentiment: bullish (from 83% bullish sentiment)
    - RSI: 58
    - Trend: neutral
    - Momentum: positive
    - Volatility: 45.0%
    - VaR (95%): 12.5%
    
    User risk tolerance: medium
    Investment amount: $10,000
    
    Task: Generate comprehensive BUY/SELL/HOLD/AVOID recommendation.
    "
)
Response: [Comprehensive recommendation with BUY/SELL/HOLD, allocation, entry strategy, etc.]

**Example 5: Multi-Turn Context**

Turn 1: "What's Bitcoin's price?" → Market Intelligence → "$92,503"
Turn 2: "Is it bullish?" → Forecasting (knows Bitcoin from context) → "Neutral trend, mixed signals"
Turn 3: "What's the risk?" → Risk & Portfolio (Bitcoin) → "45% volatility, 12.5% VaR"
Turn 4: "Should I buy?" → Synthesis (use all previous data) → "HOLD recommendation due to neutral signals"

---

**FINAL REPORT FORMAT**

When user requests comprehensive analysis or report:

**Step 1: Gather all necessary data**
Call Market Intelligence, Forecasting, and Risk agents to collect complete data.

**Step 2: Call Synthesis Agent for final recommendation**
Pass all collected data to Synthesis Agent asking for full recommendation.

**Step 3: Format Synthesis output into markdown report**
Take the Synthesis Agent's recommendation and structure it into this markdown format:
```markdown
# Investment Analysis Report: [Asset Name]

## Executive Summary
[Extract from Synthesis Agent's recommendation - 2-3 sentences with key recommendation]

## Market Intelligence
- Current Price: $X [from Market Intelligence Agent]
- Market Sentiment: X% bullish [from Market Intelligence Agent]
- Market Cap Rank: #X [from Market Intelligence Agent]

## Technical Analysis
- Trend: [from Forecasting Agent]
- RSI: X (interpretation) [from Forecasting Agent]
- Momentum: [from Forecasting Agent]
- 7-day / 30-day returns: [from Forecasting Agent]

## Risk Assessment
- Risk Score: X/100 (Risk Level) [from Synthesis Agent's assessment]
- Volatility: X% annualized [from Risk Agent]
- 95% VaR: X% [from Risk Agent]
- Suitability: [from Synthesis Agent]

## Investment Recommendation
[Extract from Synthesis Agent - includes:]
- **Action:** BUY/SELL/HOLD/AVOID
- **Confidence:** High/Medium/Low
- **Reasoning:** [Why this recommendation]
- **Allocation:** X-X% of portfolio
- **Entry Strategy:** [DCA timeline or lump sum]
- **Price Targets:** Short-term $X, Medium-term $X
- **Stop Loss:** $X (X% below entry)

## Key Risks & Considerations
[Extract from Synthesis Agent's risk warnings]

## Action Items
[Extract from Synthesis Agent or create based on recommendation:]
1. [Specific step]
2. [Specific step]
3. [Specific step]

## Conclusion
[Extract from Synthesis Agent - final 2-3 sentence verdict]

---
*Report generated: [Current date]*
*Data sources: Market Intelligence, Technical Analysis, Risk Assessment, Investment Synthesis*
```

**Note:** You are assembling the report from agent outputs, not generating new content. The Synthesis Agent provides the recommendation/reasoning, you structure it into report format.

---

**RULES**

1. **Respect boundaries** - Don't ask agents to do tasks outside their scope
2. **All agents independent** - No cross-agent data coordination needed (except Synthesis)
3. **Parse carefully for Synthesis** - Extract exact numbers and strings from agent outputs
4. **Maintain context** - Remember previous turns in conversation
5. **For complex queries** - State your plan before executing
6. **Be efficient** - Answer simple questions directly without calling agents
7. **Synthesize well** - Combine insights, don't just relay outputs

**You are the conductor. Make the orchestra play together beautifully.**

**HUMAN-IN-THE-LOOP (HITL) - PLAN APPROVAL**

For complex queries requiring multiple agents (3+ agent calls or investment recommendations):

**Step 1: Develop a plan FIRST**
Before calling any agents, outline your plan:
1. What data you need
2. Which agents you'll call
3. In what order
4. What the final output will be

**Step 2: Present plan to user for approval**
Format: "Here's my plan to answer your question:
1. [Step 1]
2. [Step 2]
3. [Step 3]

Does this plan look good? Reply 'yes' or 'proceed' to execute, or suggest modifications."

**Step 3: Wait for approval**
Do NOT call any agents until user approves with words like:
- "yes", "proceed", "go ahead", "execute", "looks good", "approved"

**Step 4: Execute plan after approval**
Once approved, execute the plan and call agents as outlined.

**Example:**

User: "Should I invest $10,000 in Bitcoin? I'm a moderate investor."

Orchestrator: "Here's my plan to generate a comprehensive investment recommendation:

1. **Market Intelligence Agent**: Get Bitcoin's current price and market sentiment
2. **Forecasting & Technical Analyst Agent**: Analyze technical indicators (RSI, trend, momentum)
3. **Risk & Portfolio Agent**: Calculate Bitcoin's volatility and Value-at-Risk for $10,000
4. **Synthesis & Recommendation Agent**: Generate BUY/SELL/HOLD recommendation based on all data, considering your moderate risk tolerance

This will give you a complete investment analysis with risk assessment and actionable recommendation.

Does this plan look good? Reply 'yes' to proceed or suggest any modifications."

User: "yes"

Orchestrator: [Proceeds to call agents as planned]

---

**When HITL is NOT needed (simple queries):**
- Single agent queries: "What's Bitcoin's price?" → Just call Market Intelligence
- Simple 2-agent queries: "What's Bitcoin's RSI?" → Just call Forecasting
- User explicitly says "just do it" or similar

**When HITL IS needed:**
- Investment recommendations ("Should I buy?")
- Portfolio analysis (multiple coins, risk assessment)
- 3+ agent coordination
- Queries involving user's money/risk tolerance

"""

# INTIIALIZE TEAM OF SUB AGENTS - share the same base LLM
sub_agents_shared_llm = OpenAILLMGenAIHub(model_name='gpt-4o', temperature=0.)

market_analyst_agent = MarketAnalystAgent(llm=sub_agents_shared_llm)
forecasting_analyst_agent = ForecastingTechnicalAnalystAgent(llm=sub_agents_shared_llm)
risk_portfolio_agent = RiskPortfolioAgent(llm=sub_agents_shared_llm)
synthesis_recco_agent = SynthesisReccomendationAgent(llm=sub_agents_shared_llm)


# EXPOSE SUB AGENTS AS FUNCTIONS THE ORCHESTRATOR CAN CALL
def execute_market_analyst_agent_tasks(request: str) -> str:
    """Execute market intelligence and data retrieval tasks using natural language.

    Use this to fetch current cryptocurrency market data, prices, sentiment, and trending information.
    This agent provides real-time market context but does NOT perform analysis, risk assessment, 
    or make recommendations.

    **When to use:**
    - Getting current prices or market data for specific coins
    - Checking what cryptocurrencies are trending
    - Retrieving market sentiment indicators
    - Fetching basic market information (market cap rank, trading volume, etc.)

    **Inputs this agent needs:**
    - Cryptocurrency name or symbol (e.g., "Bitcoin", "BTC", "Ethereum")
    - Specific data requested (price, market data, trending list, sentiment)

    **Example requests:**
    - "What is Bitcoin's current price?"
    - "Get Ethereum's market sentiment and trading volume"
    - "Show me the top 5 trending coins today"
    - "What's the market cap rank of Solana?"

    **Returns:** Current market data and information (does not include technical analysis or recommendations)
    """
    print(f"\n{'='*60}")
    print(f"🔍 SUB-AGENT: {market_analyst_agent.name}")
    print(f"{'='*60}")
    
    result = market_analyst_agent.invoke([HumanMessage(content=request)])
    
    print(f"\n{'='*60}")
    print(f"✓ SUB-AGENT COMPLETE")
    print(f"{'='*60}\n")
    
    return result["messages"][-1].content

def execute_forecasting_analyst_agent_tasks(request: str) -> str:
    """Execute technical analysis and forecasting tasks using natural language.

    Use this to fetch historical_close_prices_and_volumes,
    analyze price trends, calculate technical indicators, assess momentum, 
    and evaluate market timing for cryptocurrencies. This agent provides technical analysis 
    but does NOT calculate risk metrics or make investment recommendations.

    **When to use:**
    - Calculating technical indicators (RSI, moving averages, EMA)
    - Analyzing price trends and momentum
    - Determining if an asset is bullish, bearish, or neutral
    - Evaluating recent price performance (7-day, 30-day returns)
    - Assessing market timing based on technical signals
    - Comparing technical indicators across multiple cryptocurrencies

    **Inputs this agent needs:**
    - Cryptocurrency name or symbol
    - Type of analysis requested (RSI, trend analysis, momentum, comparison)
    - Time period if relevant (default: 60 days for accurate indicators)

    **Example requests:**
    - "What's Bitcoin's RSI right now?"
    - "Is Ethereum showing a bullish or bearish trend?"
    - "Give me a complete technical analysis of Solana"
    - "Compare the technical indicators of Bitcoin and Ethereum"
    - "What's the 7-day and 30-day return for BTC?"

    **Returns:** Technical indicators, trend signals, momentum analysis, and price patterns 
    (does not include risk metrics or investment recommendations)
    """
    print(f"\n{'='*60}")
    print(f"🔍 SUB-AGENT: {forecasting_analyst_agent.name}")
    print(f"{'='*60}")
    
    result = forecasting_analyst_agent.invoke([HumanMessage(content=request)])
    
    print(f"\n{'='*60}")
    print(f"✓ SUB-AGENT COMPLETE")
    print(f"{'='*60}\n")
    
    return result["messages"][-1].content

def execute_risk_portfolio_agent_tasks(request: str) -> str:
    """Execute risk assessment and portfolio analysis tasks using natural language.

    Use this to calculate portfolio volatility, assess downside risk (VaR), analyze 
    diversification through correlations, and evaluate portfolio health. This agent 
    provides risk metrics but does NOT fetch price data or make investment recommendations.

    **When to use:**
    - Calculating portfolio volatility and risk levels
    - Assessing Value-at-Risk (VaR) and downside risk
    - Analyzing correlation between portfolio assets
    - Evaluating portfolio diversification quality
    - Checking if portfolio allocation is balanced
    - Identifying concentration risks

    **Inputs this agent needs:**
    - Portfolio holdings with weights (must sum to 1.0)
    - Cryptocurrency names/symbols for analysis
    - Historical price data (will request from Forecasting Agent if needed)
    - Portfolio value if calculating specific VaR amounts

    **Example requests:**
    - "How risky is a portfolio with 60% Bitcoin and 40% Ethereum?"
    - "Calculate the Value-at-Risk for $10,000 invested in Solana"
    - "What's the correlation between BTC, ETH, and SOL?"
    - "Is my portfolio of 50% BTC, 30% ETH, 20% SOL balanced?"
    - "Analyze the diversification of my current holdings"

    **Returns:** Volatility metrics, VaR calculations, correlation analysis, and diversification 
    scores (does not include price data or investment recommendations)
    """
    print(f"\n{'='*60}")
    print(f"🔍 SUB-AGENT: {risk_portfolio_agent.name}")
    print(f"{'='*60}")
    
    result = risk_portfolio_agent.invoke([HumanMessage(content=request)])
    
    print(f"\n{'='*60}")
    print(f"✓ SUB-AGENT COMPLETE")
    print(f"{'='*60}\n")
    
    return result["messages"][-1].content

def execute_synthesis_recommendation_agent_tasks(request: str) -> str:
    """Execute investment synthesis and recommendation generation using natural language.

    Use this to generate comprehensive risk scores, produce actionable investment recommendations 
    (BUY/SELL/HOLD/AVOID), and synthesize findings from other agents into cohesive investment 
    guidance. This is the "senior analyst" that coordinates all other agents.

    **When to use:**
    - Generating overall risk assessments (0-100 risk scores)
    - Creating BUY/SELL/HOLD/AVOID recommendations
    - Synthesizing technical, market, and risk data into actionable advice
    - Providing investment strategies (entry/exit, allocation, stop-loss)
    - Matching investments to user's risk tolerance
    - Explaining complex risk metrics in plain language

    **Inputs this agent needs:**
    - Data from Market Intelligence Agent (current price, sentiment)
    - Data from Forecasting Agent (technical indicators, trends, momentum)
    - Data from Risk & Portfolio Agent (volatility, VaR, correlations)
    - User's risk tolerance (low/medium/high)
    - Investment amount (if applicable)

    **Example requests:**
    - "Should I buy Bitcoin? I have medium risk tolerance"
    - "Generate an investment recommendation for Ethereum with $5,000"
    - "What's the overall risk score for investing in Solana?"
    - "Analyze my portfolio of 60% BTC, 40% ETH and recommend actions"
    - "Should I invest $10,000 in crypto? I'm a conservative investor"

    **Returns:** Comprehensive risk scores with explanations, BUY/SELL/HOLD/AVOID recommendations 
    with reasoning, allocation suggestions, entry/exit strategies, price targets, stop-loss levels, 
    and risk warnings
    """
    print(f"\n{'='*60}")
    print(f"🔍 SUB-AGENT: {synthesis_recco_agent.name}")
    print(f"{'='*60}")
    
    result = synthesis_recco_agent.invoke([HumanMessage(content=request)])
    
    print(f"\n{'='*60}")
    print(f"✓ SUB-AGENT COMPLETE")
    print(f"{'='*60}\n")
    
    return result["messages"][-1].content


TOOLS = [
    PythonTool(execute_market_analyst_agent_tasks),
    PythonTool(execute_forecasting_analyst_agent_tasks),
    PythonTool(execute_risk_portfolio_agent_tasks),
    PythonTool(execute_synthesis_recommendation_agent_tasks),
]

class OrchestratorAgent(Agent):
    def __init__(self, llm, name=NAME, tools=TOOLS, system_prompt=SYSTEM_PROMPT):
        super().__init__(name, llm, tools, system_prompt)