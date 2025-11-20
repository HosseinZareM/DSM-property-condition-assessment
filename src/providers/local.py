import requests
import json
import base64
from src.providers.base import BaseVLM
from src.config import Config

class LocalVLM(BaseVLM):
    def __init__(self, model_name=Config.MODEL_LOCAL):
        super().__init__(model_name)
        self.api_url = f"{Config.OLLAMA_BASE_URL}/api/generate"

    def analyze(self, image_path, prompt):
        # Encode image
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "format": "json"  # Enforce JSON output for structured data
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error calling Local VLM: {e}")
            return None

