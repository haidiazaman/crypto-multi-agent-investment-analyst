from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from src.models.base import BaseLLM


class OpenAILLM(BaseLLM):
    """OpenAI provider - tool calls already properly formatted"""
    
    def _initialize_model(self):
        return ChatOpenAI(model=self.model_name, temperature=self.temperature)
    
    def parse_tool_calls(self, response: AIMessage) -> AIMessage:
        """OpenAI responses already have tool_calls populated correctly"""
        return response
