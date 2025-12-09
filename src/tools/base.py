from typing import Any
from abc import ABC, abstractmethod


class AgentTool(ABC):
    """Abstract base class for all agent tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool"""
        pass
    
    @abstractmethod
    def to_langchain_tool(self):
        """Convert to LangChain tool format"""
        pass

