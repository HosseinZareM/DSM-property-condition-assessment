import os
import base64
import requests
from src.providers.base import BaseVLM
from src.config import Config
from src.data_loader import DataLoader

class TogetherVLM(BaseVLM):
    def __init__(self, model_name=Config.MODEL_TOGETHER):
        super().__init__(model_name)
        self.api_key = Config.TOGETHER_API_KEY
        self.url = "https://api.together.xyz/v1/chat/completions"

    def analyze(self, image_path, prompt):
        # Together AI Llama Vision requires a specific format
        # Note: Implementation details for Together's Vision API might vary, 
        # this follows their standard chat completion with image support pattern.
        
        # We need to verify if the specific model supports local file upload or URL only.
        # Assuming standard OpenAI-compatible format for Vision which Together often supports.
        
        base64_image = DataLoader.encode_image(image_path)
        if not base64_image:
            return "Error: Image not found"

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1,
            "stop": ["<|eot_id|>"]
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling Together AI: {e}")
            # Print detailed error if available
            try:
                print(response.text)
            except:
                pass
            return None

