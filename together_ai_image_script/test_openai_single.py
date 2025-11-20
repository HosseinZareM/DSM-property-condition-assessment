#!/usr/bin/env python3
"""
Test OpenAI API with a single image to check if it's working
"""

import requests
import base64
import json
import os
from dotenv import load_dotenv

# Load environment variables from config.env
load_dotenv('config.env')

# OpenAI API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_openai_api():
    """Test OpenAI API with a single image"""
    
    # Use the first test image
    image_path = "/home/exouser/DSM-property-condition-assessment/test_images/_3_3665.jpg"
    
    if not os.path.exists(image_path):
        print("❌ Test image not found!")
        return
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare the request payload
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = """Analyze this property image and provide a DSM score from 1-5 where:
1 = Very Healthy
2 = Healthy House  
3 = In-Between
4 = Slipping
5 = Unhealthy House

Provide your response as: SCORE: [1-5]"""

    payload = {
        "model": "gpt-5-pro-2025-10-06",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100,
        "temperature": 0.3
    }
    
    try:
        print("Testing OpenAI API...")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Model: gpt-5-pro-2025-10-06")
        print("-" * 50)
        
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                ai_response = result['choices'][0]['message']['content']
                print("✅ Success! OpenAI Response:")
                print(ai_response)
            else:
                print("❌ No choices in response")
                print(json.dumps(result, indent=2))
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_openai_api()
