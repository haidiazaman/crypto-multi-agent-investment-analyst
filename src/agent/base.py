import time
import asyncio
from typing import List, Literal
from langchain_core.messages import SystemMessage, ToolMessage, BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from src.models.base import BaseLLM
from src.tools.base import AgentTool


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
    
    def invoke(self, messages: List[BaseMessage], stream_mode: str = "values"):
        """Run the agent"""
        return self.graph.invoke({"messages": messages}, stream_mode=stream_mode)
    
    def _stream_final_response(self, content: str):
        """Stream text content word by word"""
        words = content.split()
        print("\n")
        for word in words:
            print(word, end=" ", flush=True)
            time.sleep(0.005)  # Adjust speed (0.05s = 20 words/sec)
        print("\n")  # Newline at end

    def stream(self, messages: List[BaseMessage], stream_mode: str = "values"):
        """Run the agent in streaming mode"""
        
        # TODO: but can we see what the agent is thinking before the first Tool call?

        print("\n")
        tool_call_id_mapping = {} # id: name --> for functions
        previous_len = len(messages)  # Track how many messages we've seen
        for event in self.graph.stream({"messages": messages}, stream_mode=stream_mode):
            current_messages = event["messages"]
            new_messages = current_messages[previous_len:] # Get only NEW messages since last event
            previous_len = len(current_messages)
            
            for message in new_messages:
                if isinstance(message, AIMessage):
                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_call_id_mapping[tool_call["id"]] = tool_call['name']
                            print(f"🔧 testing 123 Calling: {tool_call['name']}, Args: {str(tool_call['args'])[:100]}...")                    
                    elif message.content:
                        print(f"\n💬 {message.content}")
                        # self._stream_final_response(message.content)
                
                # elif isinstance(message, ToolMessage):
                #     tool_id = message.tool_call_id
                #     function_name = tool_call_id_mapping[tool_id]
                #     print(f"output of {function_name}: {message.content}")

    async def astream(self, messages: List[BaseMessage]):
        """
        Async stream
        Run the agent in streaming mode with token-level streaming
        To run you need to use asyncio --> asyncio.run(agent.astream(messages))
        """
        collected_content = ""
        async for event in self.graph.astream_events({"messages": messages}, version="v2"):
            kind = event["event"]
            
            # Tool calls
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    # Tool call detected
                    for tc in chunk.tool_calls:
                        if tc['name']:
                            print(f"\n🔧 Calling: {tc['name']}\n")
                
                elif chunk.content:
                    # Stream final response token by token
                    print(chunk.content, end="", flush=True)
                    collected_content += chunk.content  # ← Collect
        
        return collected_content  # ← Return

    def conversation(self):
        """Run synchronous conversation loop with memory and streaming"""
        messages: list[BaseMessage] = []
        
        while True:
            user_input = input("\n❓ ask me smth: ")
            if user_input.lower() in ["/bye", "exit", "quit"]:
                print("\n👋 End of conversation, bye!")
                break
            
            messages.append(HumanMessage(content=user_input))
            self.stream(messages=messages)
            print()  # Newline after response

    async def aconversation(self):
        """Run async conversation loop with memory and streaming"""
        messages: list[BaseMessage] = []
        
        while True:
            user_input = input("\n❓ ask me smth: ")
            if user_input.lower() in ["/bye", "exit", "quit"]:
                print("\n👋 End of conversation, bye!")
                break
            
            messages.append(HumanMessage(content=user_input))
            response = await self.astream(messages=messages)
            messages.append(AIMessage(content=response))      # ← Add to history
            print()  # Newline after response



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