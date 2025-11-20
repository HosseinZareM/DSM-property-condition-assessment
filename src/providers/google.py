import base64
import PIL.Image
from src.providers.base import BaseVLM
from src.config import Config

# Try to import the new google-genai package first (if user has it), fallback to google-generativeai
try:
    from google import genai
    from google.genai import types
    HAS_NEW_GOOGLE_API = True
except ImportError:
    try:
        import google.generativeai as genai
        HAS_NEW_GOOGLE_API = False
    except ImportError:
        genai = None
        HAS_NEW_GOOGLE_API = None
        print("Warning: Google Generative AI package not found. Install with: pip install google-generativeai")

class GoogleVLM(BaseVLM):
    def __init__(self, model_name=Config.MODEL_GOOGLE):
        super().__init__(model_name)
        
        if genai is None:
            raise ImportError("Google Generative AI package not found. Install with: pip install google-generativeai or pip install google-genai")
        
        if HAS_NEW_GOOGLE_API:
            # New API format (google-genai package)
            self.client = genai.Client(
                api_key=Config.GOOGLE_API_KEY,
                http_options={'api_version': 'v1alpha'}
            )
            self.use_new_api = True
        else:
            # Standard API format (google-generativeai package)
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(model_name)
            self.use_new_api = False

    def analyze(self, image_path, prompt):
        try:
            if self.use_new_api:
                # New API format
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        types.Content(
                            parts=[
                                types.Part(text=prompt),
                                types.Part(
                                    inline_data=types.Blob(
                                        mime_type="image/jpeg",
                                        data=image_bytes,
                                    )
                                )
                            ]
                        )
                    ]
                )
                return response.text
            else:
                # Standard API format
                img = PIL.Image.open(image_path)
                response = self.model.generate_content([prompt, img])
                return response.text
        except Exception as e:
            print(f"Error calling Google Gemini: {e}")
            return None
