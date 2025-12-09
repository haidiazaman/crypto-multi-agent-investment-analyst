from langchain_core.messages import SystemMessage, ToolMessage, BaseMessage, HumanMessage, AIMessage
from src.agent.base import Agent
from src.models.ollama_model import OllamaLLM
from src.tools.crypto_tools import get_price, get_coin_list, get_coin_market_data
from src.tools.arithmethic_tools import add, multiply, subtract, divide
from src.tools.python_tool import PythonTool
# from langchain.tools import tool
from langchain_core.messages import HumanMessage


def create_crypto_agent():
    # CRYPTO_AGENT
    NAME = "CryptoAgent"
    MODEL_KEY = "qwen2.5:14b-instruct"
    SYSTEM_PROMPT = """
    You are an expert at cryptocurrency with access to some crypto tools.
    Think and respond only in English.
    """
    # Create tool instances
    tools = [
        PythonTool(get_price),
        PythonTool(get_coin_list),
        PythonTool(get_coin_market_data),  
    ]
    # Create agent with Ollama
    ollama_llm = OllamaLLM(model_name=MODEL_KEY)
    agent = Agent(name=NAME, llm=ollama_llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    return agent

def create_math_agent():
    # MATH_AGENT
    NAME = "MathAgent"
    MODEL_KEY = "qwen2.5:14b-instruct"
    SYSTEM_PROMPT = """
    You are an agent who has access to different arithmetic tools. Your only job is to do arithmetic using only the tools provided to you.
    Think and respond only in English.
    """
    # Create tool instances
    tools = [
        PythonTool(add),
        PythonTool(subtract),
        PythonTool(multiply),  
        PythonTool(divide),  
    ]
    # Create agent with Ollama
    ollama_llm = OllamaLLM(model_name=MODEL_KEY)
    agent = Agent(name=NAME, llm=ollama_llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    return agent

crypto_agent = create_crypto_agent()
math_agent = create_math_agent()

# @tool
def execute_crypto_tasks(request: str) -> str:
    """Execute cryptocurrency-related tasks using natural language.

    Use this when the user wants to analyze, check, or compute anything related
    to cryptocurrencies. Handles market data queries, price lookups, portfolio
    calculations, risk evaluation, and general crypto reasoning.

    Input: Natural language crypto request (e.g., 'what is BTC's support level?',
    'calculate my PnL if ETH goes to 3500', 'compare SOL vs AVAX fundamentals')
    """
    result = crypto_agent.invoke([
        HumanMessage(content=request)
    ])
    print("\n### CRYPTO AGENT Intermediate Steps ###")
    new_messages = result["messages"]

    for msg in new_messages[:-1]:  # All except the last (final response)
        if isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"🔧 Calling tool: {msg.tool_calls[0]['name']}")
            print(f"   Args: {msg.tool_calls[0]['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"📊 Tool result: {msg.content[:100]}...")  # Truncate long results
    print("### End Intermediate Steps ###\n")
    return result["messages"][-1].text


# @tool
def execute_arithmetic_tasks(request: str) -> str:
    """Perform arithmetic operations using natural language.

    Use this when the user wants to compute, evaluate, or simplify any expression
    involving numbers. Handles addition, subtraction, multiplication, division,
    multi-step calculations, and basic word-problem reasoning.

    Input: Natural language arithmetic request (e.g., 'take 42, multiply by 7,
    subtract 19, and divide by 3')
    """
    result = math_agent.invoke([
        HumanMessage(content=request)
    ])
    print("\n### MATH AGENT Intermediate Steps ###")
    new_messages = result["messages"]

    for msg in new_messages[:-1]:  # All except the last (final response)
        if isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"🔧 Calling tool: {msg.tool_calls[0]['name']}")
            print(f"   Args: {msg.tool_calls[0]['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"📊 Tool result: {msg.content[:100]}...")  # Truncate long results
    print("### End Intermediate Steps ###\n")
    return result["messages"][-1].text


def create_supervisor_agent():
    NAME = "SupervisorAgent"
    MODEL_KEY = "qwen2.5:14b-instruct"
    SYSTEM_PROMPT = """
    You are the Supervisor Agent.

    Your job:
    - Decide whether the user query requires MathAgent or CryptoAgent.
    - If needed, call the correct agent tool.
    - Otherwise, answer directly.

    Rules:
    - Think step-by-step.
    - Use tools when appropriate.

    Respond and think only in English
    """
    
    # Create tool instances
    tools = [
        PythonTool(execute_arithmetic_tasks),
        PythonTool(execute_crypto_tasks),
    ]
    # Create agent with Ollama
    ollama_llm = OllamaLLM(model_name=MODEL_KEY)
    agent = Agent(name=NAME, llm=ollama_llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    return agent
