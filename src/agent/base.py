from typing import List, Literal
from langchain_core.messages import SystemMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from src.models.base import BaseLLM
from src.tools.base import AgentTool

# ============================================================================
# Agent Class
# ============================================================================

class Agent:
    """
    Agent with pluggable LLM and tools
    
    Responsibilities:
    - Bind tools to the model (once at initialization)
    - Orchestrate the LLM → Tool → LLM loop
    - Use LLM's parsing logic for tool calls
    """
    
    def __init__(self, name: str, llm: BaseLLM, tools: List[AgentTool], system_prompt: str):
        self.name = name
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        
        # Convert tools to LangChain format
        self.langchain_tools = [tool.to_langchain_tool() for tool in tools]
        self.tools_by_name = {t.name: t for t in tools}
        
        # Bind tools to model once at initialization
        if self.langchain_tools:
            self._model_with_tools = llm._model.bind_tools(self.langchain_tools)
        else:
            self._model_with_tools = llm._model
        
        # Build the agent graph
        self.graph = self._build_graph()
    
    def _llm_call(self, state: dict):
        """LLM node - invokes model and parses tool calls"""
        messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
        
        response = self._model_with_tools.invoke(messages)
        parsed_response = self.llm.parse_tool_calls(response)
        
        return {"messages": [parsed_response]}
    
    def _tool_node(self, state: dict):
        """Tool execution node"""
        result = []
        last_message = state["messages"][-1]
        
        for tool_call in last_message.tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            observation = tool.execute(**tool_call["args"])
            result.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"]
                )
            )
        return {"messages": result}
    
    def _should_continue(self, state: MessagesState) -> Literal["tool_node", END]:
        """Routing logic: continue to tools or end"""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tool_node"
        return END
    
    def _build_graph(self):
        """Build the LangGraph agent"""
        builder = StateGraph(MessagesState)
        
        # Add nodes
        builder.add_node("llm_call", self._llm_call)
        builder.add_node("tool_node", self._tool_node)
        
        # Add edges
        builder.add_edge(START, "llm_call")
        builder.add_conditional_edges(
            "llm_call",
            self._should_continue,
            ["tool_node", END]
        )
        builder.add_edge("tool_node", "llm_call")
        
        return builder.compile()
    
    def invoke(self, messages: List[BaseMessage]):
        """Run the agent"""
        return self.graph.invoke({"messages": messages})


if __name__ == "__main__":
    #   TEST SIMPLE MATH AGENT

    from langchain_core.messages import HumanMessage
    from src.tools.python_tool import PythonTool
    from src.tools.math_tools import add, multiply, divide, subtract
    from src.models.ollama_model import OllamaLLM

    NAME = "MathAgent"
    MODEL_KEY = "qwen2.5:14b-instruct"
    SYSTEM_PROMPT = """
    You are a helpful math assistant that performs arithmetic operations.
    Think and respond only in English.
    """
    # Create tool instances
    tools = [
        PythonTool(add),
        PythonTool(multiply),
        PythonTool(divide), 
        PythonTool(subtract), 
    ]
    
    # Create agent with Ollama
    ollama_llm = OllamaLLM(model_name=MODEL_KEY)
    agent = Agent(name=NAME, llm=ollama_llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    
    user_query = "What is 25 times 4, then add 10 to that result?"
    print("=== Math Agent with Ollama ===\n")
    result = agent.invoke([
        HumanMessage(content=user_query)
    ])
    
    for msg in result["messages"]:
        msg.pretty_print()
    
    # Example: Create agent with OpenAI (would work the same way)
    # openai_llm = OpenAILLM(model_name="gpt-4")
    # openai_agent = Agent(
    #     name="OpenAIMathAgent",
    #     llm=openai_llm,
    #     tools=tools,
    #     system_prompt="You are a helpful math assistant."
    # )
