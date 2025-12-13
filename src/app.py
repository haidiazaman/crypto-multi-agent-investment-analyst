import asyncio
import streamlit as st
from dotenv import load_dotenv
from src.models.openai_model import OpenAILLM
from src.agent.orchestrator_agent import OrchestratorAgent
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.title("💰 Multi-Agent Crypto Investment Analyst")

# How Streamlit works:
# Streamlit reruns the entire script on every interaction:
# User types message → Script reruns from top
# st.session_state.messages persists (has all history)
# Display all previous messages (from session_state)
# New message added to session_state
# Convert ALL messages to LangChain format
# Call agent with FULL conversation history

# Initialize agent once --> because the script reruns from the top at every turn
# st.session_state only persists during the session. Browser refresh = new session = history cleared.
if "agent" not in st.session_state:
    llm = OpenAILLM(model_name='gpt-4o', temperature=0.)
    st.session_state.agent = OrchestratorAgent(llm=llm)

# Initialize messages --> because the script reruns from the top at every turn
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history --> because the script reruns from the top at every turn
# Script reruns from top: Loop through st.session_state.messages → Re-displays Turn 1 messages instantly (no streaming, just markdown)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask about cryptocurrency investments..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            lc_messages = [] # Convert to LangChain messages
            for msg in st.session_state.messages: # loop thru all sessions_state messages
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                else:
                    lc_messages.append(AIMessage(content=msg["content"]))
            
            response = asyncio.run(st.session_state.agent.astream(lc_messages))
            st.markdown(response)
    
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})