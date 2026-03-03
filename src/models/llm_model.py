from typing import Optional, TypeVar
import requests
import json
from pydantic import BaseModel
from abc import ABC, abstractmethod

TBaseModel = TypeVar("TBaseModel", bound=BaseModel)

class LLMBaseModel:
    @abstractmethod
    def invoke(self, *,
               user_prompt: str,
               system_prompt: Optional[str] = None,
               schema: Optional[dict] = None,
               **kargs) -> dict:
        
        raise NotImplementedError

class OpenrouterLLMModel(LLMBaseModel):
    def __init__(self, 
                 API_KEY: str,
                 MODEL: str,
                 base_url: str):
        self.API_KEY = API_KEY
        self.MODEL = MODEL
        self.base_url = base_url
        self.system_prompt: str = None
        self.schema: dict = None
    
    def invoke(self, *,
               user_prompt: str,
               system_prompt: Optional[str] = None,
               schema: Optional[dict] = None,
               header_extras: Optional[dict] = None,
               **kargs) -> dict:
        
        messages = []
        if not system_prompt:
            if self.system_prompt:
                messages.append({ "role": 'system', "content": self.system_prompt })
        else:
            messages.append({ "role": 'system', "content": system_prompt })
        messages.append({ "role": "user", "content": user_prompt})
        
        payload = {
                "model": self.MODEL,
                "messages": messages,
                **kargs
            }
        if not schema:
            if self.schema:
                payload["response_format"] = self.schema
        else:
            payload["response_format"] = schema
        
        headers = {
            "Authorization": f"Bearer {self.API_KEY }",
        }
        if header_extras:
            headers = {**headers, **header_extras}
        
        # hit api
        response = requests.post(
            url=self.base_url,
            headers=headers,
            data=json.dumps(payload)
            )
        
        response.raise_for_status()
        return response.json()
    
    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
    
    def set_schema(self, pydantic_output_model: TBaseModel):
        self.schema = self.get_formatted_schema(pydantic_output_model.model_json_schema())
    
    def get_formatted_schema(self, schema: dict):
        return {
            "type": "json_schema",
            "json_schema": {
            "name": "response",
            "strict": True,
            "schema": schema
            }
        }