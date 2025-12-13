# from src.models.ollama_model import OllamaLLM
import asyncio
from src.agent.orchestrator_agent import OrchestratorAgent
from src.models.openai_genaihub import OpenAILLMGenAIHub


if __name__=="__main__":
    # MODEL_KEY = "qwen2.5:14b-instruct"
    # smarter_orchestrator_llm = OllamaLLM(model_name=MODEL_KEY)
    llm = OpenAILLMGenAIHub(model_name='gpt-4o', temperature=0.)
    agent = OrchestratorAgent(llm=llm)
    asyncio.run(agent.aconversation())