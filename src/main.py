# from src.models.ollama_model import OllamaLLM
from src.agent.orchestrator_agent import OrchestratorAgent
from langchain_core.messages import SystemMessage, ToolMessage, BaseMessage, HumanMessage, AIMessage

from src.models.openai_genaihub import OpenAILLMGenAIHub

# PRINT BETTER INTERMEDIATE STEPS
if __name__=="__main__":
    MODEL_KEY = "qwen2.5:14b-instruct"
    # smarter_orchestrator_llm = OllamaLLM(model_name=MODEL_KEY)
    smarter_orchestrator_llm = OpenAILLMGenAIHub(model_name='gpt-4o', temperature=0.)

    agent = OrchestratorAgent(llm=smarter_orchestrator_llm)
    
    messages: list[BaseMessage] = []

    while True:
        user_input = input("ask me smth: ")
        if user_input == "/bye":
            break
        
        messages.append(HumanMessage(user_input))
        result = agent.invoke(messages=messages)
        
        # Get new messages added in this iteration
        new_messages = result["messages"][len(messages):]
        
        # Check if tools were executed (more than 1 new message means tool was used)
        if len(new_messages) > 1:
            print("\n--- Intermediate Steps ---")
            for msg in new_messages[:-1]:  # All except the last (final response)
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    print(f"🔧 Calling tool: {msg.tool_calls[0]['name']}")
                    print(f"   Args: {msg.tool_calls[0]['args']}")
                elif isinstance(msg, ToolMessage):
                    print(f"📊 Tool result: {msg.content[:100]}...")  # Truncate long results
            print("--- End Intermediate Steps ---\n")
        
        # Update messages with full result
        messages = result["messages"]
        
        # Print final AI response
        last_message = messages[-1]
        if isinstance(last_message, AIMessage):
            print(f"Agent: {last_message.content}\n")

    print("\n end of convo, bye! 🤩")