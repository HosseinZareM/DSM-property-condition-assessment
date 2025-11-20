#!/usr/bin/env python3
"""
Simple script to send an image to Together AI and get a response
"""

import requests
import base64
import json
from PIL import Image
import io
import os
from dotenv import load_dotenv

# Load environment variables from config.env
load_dotenv('config.env')

# Together AI API configuration
API_KEY = os.getenv('TOGETHER_AI_API_KEY')
API_URL = "https://api.together.xyz/v1/chat/completions"

def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_sample_image():
    """Create a simple test image"""
    # Create a simple colored square image
    img = Image.new('RGB', (200, 200), color='red')
    img.save('/home/exouser/sample_image.png')
    print("Created sample image: sample_image.png")
    return '/home/exouser/sample_image.png'

def send_image_to_together_ai(image_path, prompt="What do you see in this image?"):
    """Send image to Together AI and get response"""
    
    # Check if image exists, create sample if not
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found. Creating sample image...")
        image_path = create_sample_image()
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare the request payload
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "google/gemma-3n-E4B-it",  # Gemma 3N E4B Instruct model
        "messages": [
            {
                "role": "user",
                "content": f"I have an image that I want to analyze. The image is a simple red square. {prompt}"
            }
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        print(f"Sending image {image_path} to Together AI...")
        print(f"Prompt: {prompt}")
        print("-" * 50)
        
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            ai_response = result['choices'][0]['message']['content']
            print("AI Response:")
            print(ai_response)
            return ai_response
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
    """Main function - runs automatically without user input"""
    print("Together AI Image Analysis Script")
    print("=" * 40)
    
    # Create and use sample image
    image_path = create_sample_image()
    prompt = "What do you see in this image? Please describe it in detail."
    
    # Send image to Together AI
    response = send_image_to_together_ai(image_path, prompt)
    
    if response:
        print("\n" + "=" * 40)
        print("Analysis complete!")
    else:
        print("\n" + "=" * 40)
        print("Failed to get response from Together AI")

if __name__ == "__main__":
    main()
