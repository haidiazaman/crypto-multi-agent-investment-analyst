import asyncio
from src.agent.base import Agent
from src.tools.python_tool import PythonTool
from src.models.openai_model import OpenAILLM
from src.agent.prompts.forecasting_analyst_prompt import SYSTEM_PROMPT, EXECUTE_FUNCTION_DESCRIPTION
from src.tools.forecasting_analysis_tools import get_historical_close_prices_and_volumes, calculate_technical_indicators, analyze_price_volume_trend


NAME = "Forecasting & Technical Analysis Agent"
TOOLS = [
    PythonTool(get_historical_close_prices_and_volumes),
    PythonTool(calculate_technical_indicators),
    PythonTool(analyze_price_volume_trend),  
]

class ForecastingTechnicalAnalystAgent(Agent):
    DESCRIPTION = EXECUTE_FUNCTION_DESCRIPTION

    def __init__(self, llm, name=NAME, tools=TOOLS, system_prompt=SYSTEM_PROMPT):
        super().__init__(name, llm, tools, system_prompt)

if __name__=="__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    llm = OpenAILLM(model_name='gpt-4o', temperature=0.)
    agent = ForecastingTechnicalAnalystAgent(llm=llm)
    asyncio.run(agent.aconversation())