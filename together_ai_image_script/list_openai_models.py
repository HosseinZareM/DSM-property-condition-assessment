#!/usr/bin/env python3
"""
List available OpenAI models
"""

import requests
import json
from dotenv import load_dotenv

# Load environment variables from config.env
load_dotenv('config.env')

# OpenAI API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODELS_URL = "https://api.openai.com/v1/models"

def list_openai_models():
    """List all available OpenAI models"""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(MODELS_URL, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        if 'data' in result:
            print("Available OpenAI Models:")
            print("=" * 50)
            
            # Filter for vision-capable models
            vision_models = []
            for model in result['data']:
                model_id = model.get('id', '')
                if 'gpt' in model_id.lower() and ('vision' in model_id.lower() or '4o' in model_id.lower() or '5' in model_id.lower()):
                    vision_models.append(model_id)
                    print(f"✅ {model_id}")
            
            print(f"\nFound {len(vision_models)} vision-capable models")
            
            # Save to file
            with open('/home/exouser/DSM-property-condition-assessment/together_ai_image_script/logs/openai_models.txt', 'w') as f:
                f.write("Available OpenAI Models:\n")
                f.write("=" * 50 + "\n")
                for model in result['data']:
                    f.write(f"{model.get('id', 'Unknown')}\n")
            
            print("Full list saved to logs/openai_models.txt")
            
        else:
            print("No models found in response")
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_openai_models()


