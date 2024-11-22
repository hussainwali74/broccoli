# from g4f.client import Client
from g4f.Provider import AIChatFree , ChatgptFree
from openai import OpenAI
from pydantic import BaseModel
from typing import List
class Message(BaseModel):
    role: str
    content: str
from dotenv import load_dotenv
load_dotenv()

class LLM:
    def __init__(self, model_name='gpt-4o-mini'):
        # self.client = Client()
        self.client = OpenAI()
        self.model_name = model_name
        
    def generate_text(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            provider=AIChatFree,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def chat(self, messages: List[Message]):
        messages_dict = [message.model_dump() for message in messages]

        response = self.client.chat.completions.create(
            model=self.model_name, 
            messages=messages_dict
        )
        return response.choices[0].message.content

    def fast_generate(self, prompt):
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            provider=ChatgptFree,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def instruct_generate(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            provider=AIChatFree,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content