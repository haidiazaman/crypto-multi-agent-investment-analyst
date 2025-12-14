import asyncio
from src.agent.base import Agent
from src.tools.python_tool import PythonTool
from src.models.openai_model import OpenAILLM
from src.agent.prompts.market_intelligence_analyst_prompt import SYSTEM_PROMPT, EXECUTE_FUNCTION_DESCRIPTION
from src.tools.market_intelligence_tools import get_current_coin_price, get_current_coin_market_data, get_current_trending_coins


NAME = "Market Intelligence Analyst Agent"
TOOLS = [
    PythonTool(get_current_coin_price),
    PythonTool(get_current_coin_market_data),
    PythonTool(get_current_trending_coins),  
]

class MarketAnalystAgent(Agent):
    DESCRIPTION = EXECUTE_FUNCTION_DESCRIPTION
    
    def __init__(self, llm, name=NAME, tools=TOOLS, system_prompt=SYSTEM_PROMPT):
        super().__init__(name, llm, tools, system_prompt)

if __name__=="__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    llm = OpenAILLM(model_name='gpt-4o', temperature=0.)
    agent = MarketAnalystAgent(llm=llm)
    asyncio.run(agent.aconversation())