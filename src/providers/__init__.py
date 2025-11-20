from src.providers.local import LocalVLM
from src.providers.openai import OpenAIVLM
from src.providers.google import GoogleVLM
from src.providers.together import TogetherVLM
from src.config import Config

def get_provider(provider_name):
    if provider_name == "local":
        return LocalVLM()
    elif provider_name == "openai":
        return OpenAIVLM()
    elif provider_name == "google":
        return GoogleVLM()
    elif provider_name == "together":
        return TogetherVLM()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

