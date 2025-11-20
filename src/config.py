import os
from dotenv import load_dotenv

# Load .env file only from src directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

# Load .env file from src directory only
if os.path.exists(ENV_FILE_SRC):
    load_dotenv(ENV_FILE_SRC)
else:
    # Fallback to default behavior (current directory) if src/.env doesn't exist
    load_dotenv()

class Config:
    # Paths - Support both old "Data" structure and new "data" structure
    
    # Check if Data folder exists (old structure) or data folder (new structure)
    if os.path.exists(os.path.join(BASE_DIR, "Data")):
        DATA_DIR = "Data"
        RAW_IMAGES_DIR = os.path.join(DATA_DIR, "extractedimages")
        ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotation")
    else:
        DATA_DIR = "data"
        RAW_IMAGES_DIR = os.path.join(DATA_DIR, "extractedimages")
        ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotation")
    
    PROMPTS_DIR = os.path.join("data", "prompts")
    OUTPUTS_DIR = os.path.join("data", "outputs")

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    # Support both TOGETHER_API_KEY and TOGETHER_AI_API_KEY for compatibility
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY") or os.getenv("TOGETHER_AI_API_KEY")

    # Model Names - Updated to Latest Versions (2024-2025)
    
    # OpenAI: GPT-4o is the latest vision model
    # Note: GPT-5.1 not yet released as of 2024 - GPT-4o is the current latest vision model
    # If GPT-5.1 becomes available, update to "gpt-5.1" or "gpt-5.1-vision"
    MODEL_OPENAI = "gpt-4o"  # Latest vision model (as of 2024)
    # Alternatives: "gpt-4-turbo", "gpt-4o-2024-08-06"
    
    # Google: Gemini 3 Pro Preview (latest)
    MODEL_GOOGLE = "gemini-3-pro-preview"  # Latest Gemini 3 model with vision
    # Alternatives if preview not available: "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-pro-latest"
    
    # Together AI: Vision-capable model
    MODEL_TOGETHER = "Qwen/Qwen2.5-VL-72B-Instruct"  # Qwen 2.5 Vision Large model (72B)
    # Alternatives: "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
    
    # Local (Ollama): Latest vision model
    # Note: Update this to match your installed Ollama model
    MODEL_LOCAL = "gemma3:27b"  # Currently installed model (supports vision)
    # To install other vision models, run: ollama pull llama3.2-vision
    # Alternatives: "llama3.2-vision", "llama3.1:8b-vision", "llava:latest", "gemma2:9b-vision"

    # Settings
    OLLAMA_BASE_URL = "http://localhost:11434"

