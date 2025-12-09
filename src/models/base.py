from typing import List
from abc import ABC, abstractmethod
from langchain_core.messages import AIMessage, BaseMessage


class BaseLLM(ABC):
    """
    Base class for LangChain chat models (ChatOpenAI, ChatOllama, etc.)
    
    Responsibilities:
    - Initialize the underlying LangChain chat model
    - Provide invoke method that returns AIMessage
    - Parse tool calls from provider-specific formats
    """
    
    def __init__(self, model_name: str, temperature: float = 0):
        self.model_name = model_name
        self.temperature = temperature
        self._model = self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize and return a LangChain BaseChatModel instance"""
        pass
    
    @abstractmethod
    def parse_tool_calls(self, response: AIMessage) -> AIMessage:
        """Parse tool calls from response (provider-specific logic)"""
        pass
    
    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """Invoke the underlying chat model"""
        return self._model.invoke(messages)