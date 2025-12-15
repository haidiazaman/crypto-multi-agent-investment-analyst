from typing import Any
from src.tools.base import AgentTool
from langchain_core.tools import tool


class PythonTool(AgentTool):
    """Wrapper for raw Python functions"""
    
    def __init__(self, func: callable):
        name = func.__name__
        description = func.__doc__ or f"Execute {name}"
        super().__init__(name, description)
        self.func = func
    
    def execute(self, **kwargs) -> Any:
        return self.func(**kwargs)
    
    def to_langchain_tool(self):
        return tool(self.func)