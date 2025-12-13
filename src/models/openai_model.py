from src.models.base import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage


class OpenAILLM(BaseLLM):
    """OpenAI provider - tool calls already properly formatted"""
    
    def _initialize_model(self):
        return ChatOpenAI(model=self.model_name, temperature=self.temperature)
    
    def parse_tool_calls(self, response: AIMessage) -> AIMessage:
        """OpenAI responses already have tool_calls populated correctly"""
        return response

if __name__=="__main__":
    from dotenv import load_dotenv
    load_dotenv()

    model = OpenAILLM(model_name="gpt-4o", temperature=0.)
    message = "what is the tallest mountain in the world"

    resp = model.invoke(messages=message)
    print(resp.content)