#!/usr/bin/env python3
"""
Script to list all available models from Together AI
"""

import requests
import json

# Together AI API configuration
API_KEY = "tgp_v1_20wpGwgQcqOZn5aaoYA_-NihgYGHUYks7i44R9AecfQ"
MODELS_URL = "https://api.together.xyz/v1/models"

def list_available_models():
    """List all available models from Together AI"""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        print("Fetching available models from Together AI...")
        print("=" * 50)
        
        response = requests.get(MODELS_URL, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        if 'data' in result:
            models = result['data']
            print(f"Found {len(models)} available models:")
            print("-" * 50)
            
            # Filter for Gemma models
            gemma_models = []
            for model in models:
                model_id = model.get('id', '')
                if 'gemma' in model_id.lower():
                    gemma_models.append(model)
                    print(f"🔍 GEMMA MODEL: {model_id}")
                    if 'name' in model:
                        print(f"   Name: {model['name']}")
                    if 'description' in model:
                        print(f"   Description: {model['description']}")
                    print()
            
            if gemma_models:
                print(f"Found {len(gemma_models)} Gemma models!")
            else:
                print("No Gemma models found.")
            
            print("\n" + "=" * 50)
            print("ALL AVAILABLE MODELS:")
            print("-" * 50)
            
            for i, model in enumerate(models, 1):
                model_id = model.get('id', 'Unknown')
                model_name = model.get('name', 'No name')
                print(f"{i:3d}. {model_id}")
                if model_name != 'No name':
                    print(f"     Name: {model_name}")
            
            return models
            
        else:
            print("Unexpected response format:")
            print(json.dumps(result, indent=2))
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def main():
    """Main function"""
    print("Together AI Models List")
    print("=" * 30)
    
    models = list_available_models()
    
    if models:
        print(f"\n✅ Successfully retrieved {len(models)} models!")
    else:
        print("\n❌ Failed to retrieve models")

if __name__ == "__main__":
    main()
