from src.agent.base import Agent
from src.tools.math_tools import add, divide, multiply, subtract
from src.tools.python_tool import PythonTool

# NOT USED FOR THE PROJECT 

NAME = "Mathematics Agent"
SYSTEM_PROMPT = """
You are an agent who has access to different arithmetic tools. Your only job is to do arithmetic using only the tools provided to you.
Think and respond only in English.
"""

TOOLS = [
    PythonTool(add),
    PythonTool(subtract),
    PythonTool(multiply),  
    PythonTool(divide),  
]

class MathsAgent(Agent):
    def __init__(self, llm, name=NAME, tools=TOOLS, system_prompt=SYSTEM_PROMPT):
        super().__init__(name, llm, tools, system_prompt)