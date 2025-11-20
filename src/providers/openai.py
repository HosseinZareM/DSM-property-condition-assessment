from openai import OpenAI
from src.providers.base import BaseVLM
from src.config import Config
from src.data_loader import DataLoader

class OpenAIVLM(BaseVLM):
    def __init__(self, model_name=Config.MODEL_OPENAI):
        super().__init__(model_name)
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    def analyze(self, image_path, prompt):
        base64_image = DataLoader.encode_image(image_path)
        if not base64_image:
            return "Error: Image not found"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return None

