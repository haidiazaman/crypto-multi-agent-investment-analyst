import asyncio
from src.agent.base import Agent
from src.tools.python_tool import PythonTool
from src.models.openai_model import OpenAILLM
from src.agent.prompts.risk_portfolio_agent_prompt import SYSTEM_PROMPT, EXECUTE_FUNCTION_DESCRIPTION
from src.tools.risk_portfolio_tools import get_historical_close_prices, calculate_correlation_matrix, calculate_portfolio_volatility, calculate_returns_from_prices, calculate_var, generate_sample_returns

NAME = "Risk & Portfolio Agent"
TOOLS = [
    PythonTool(get_historical_close_prices),
    PythonTool(calculate_returns_from_prices),
    PythonTool(calculate_portfolio_volatility),
    PythonTool(calculate_var),
    PythonTool(calculate_correlation_matrix),  
]

class RiskPortfolioAgent(Agent):
    DESCRIPTION = EXECUTE_FUNCTION_DESCRIPTION
    
    def __init__(self, llm, name=NAME, tools=TOOLS, system_prompt=SYSTEM_PROMPT):
        super().__init__(name, llm, tools, system_prompt)

if __name__=="__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    llm = OpenAILLM(model_name='gpt-4o', temperature=0.)
    agent = RiskPortfolioAgent(llm=llm)
    asyncio.run(agent.aconversation())