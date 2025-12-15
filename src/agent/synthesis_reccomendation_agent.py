import asyncio
from src.agent.base import Agent
from src.tools.python_tool import PythonTool
from src.models.openai_model import OpenAILLM
from src.agent.prompts.synthesis_reccomendation_agent_prompt import SYSTEM_PROMPT, EXECUTE_FUNCTION_DESCRIPTION
from src.tools.synthesis_reccomendation_tools import generate_investment_recommendation, generate_risk_score


NAME = "Synthesis & Recommendation Agent"
TOOLS = [
    # PythonTool(generate_risk_score),
    # PythonTool(generate_investment_recommendation)
]

class SynthesisReccomendationAgent(Agent):
    DESCRIPTION = EXECUTE_FUNCTION_DESCRIPTION

    def __init__(self, llm, name=NAME, tools=TOOLS, system_prompt=SYSTEM_PROMPT):
        super().__init__(name, llm, tools, system_prompt)


if __name__=="__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    llm = OpenAILLM(model_name='gpt-4o', temperature=0.)
    agent = SynthesisReccomendationAgent(llm=llm)
    asyncio.run(agent.aconversation())