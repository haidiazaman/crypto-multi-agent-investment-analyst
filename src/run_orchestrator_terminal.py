import asyncio
from src.models.openai_model import OpenAILLM
from src.agent.orchestrator_agent import OrchestratorAgent


if __name__=="__main__":
    from dotenv import load_dotenv
    load_dotenv()

    llm = OpenAILLM(model_name='gpt-4o', temperature=0.)
    subagent_shared_llm = OpenAILLM(model_name='gpt-4o', temperature=0.)
    agent = OrchestratorAgent(llm=llm, sub_agent_shared_llm=subagent_shared_llm)
    asyncio.run(agent.aconversation())