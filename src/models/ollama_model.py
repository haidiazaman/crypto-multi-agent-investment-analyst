import re
import json
from src.models.base import BaseLLM
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage


class OllamaLLM(BaseLLM):
    """Ollama provider with custom parsing for tool calls"""
    
    def _initialize_model(self):
        return ChatOllama(model=self.model_name, temperature=self.temperature)
    
    def parse_tool_calls(self, response: AIMessage) -> AIMessage:
        """Parse Ollama's <tool_call> text format into structured tool_calls"""
        if "<tool_call>" in response.content:
            match = re.search(
                r'<tool_call>\s*(\{.*?\})\s*</tool_call>', 
                response.content, 
                re.DOTALL
            )
            if match:
                try:
                    data = json.loads(match.group(1))
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": data["name"],
                            "args": data["arguments"],
                            "id": "call_1"
                        }]
                    )
                except json.JSONDecodeError:
                    pass
        return response
    
    def stream(self, messages):
        return self._model.stream(messages)  # LangChain returns a generator

if __name__=="__main__":
    MODEL_KEY = "qwen2.5:14b-instruct"
    TEMPERATURE = 0.
    model = OllamaLLM(model_name=MODEL_KEY, temperature=TEMPERATURE)
    user_input = "where is Mt Everest?"
    response = model.invoke(user_input)
    # response = model.stream(user_input)
    print(type(response))
    print(response)