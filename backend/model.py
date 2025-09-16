from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()
class Model:
    def __init__(self,model_provider,model_name):
        self.model_provider = model_provider
        self.model_name = model_name
    def create_model(self):
        if self.model_provider == "groq":
            return ChatGroq(model=self.model_name)
        elif self.model_provider == "openai":
            return ChatOpenAI(model_name=self.model_name)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")