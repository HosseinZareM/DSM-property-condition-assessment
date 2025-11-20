#!/usr/bin/env python3
"""
Script to randomly select a property condition assessment image and analyze it with Gemma 3N
"""

import requests
import base64
import json
import os
import random
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables from config.env
load_dotenv('config.env')

# Together AI API configuration
API_KEY = os.getenv('TOGETHER_AI_API_KEY')
API_URL = "https://api.together.xyz/v1/chat/completions"

def get_random_property_image():
    """Get a random property condition assessment image"""
    base_path = "/home/exouser/together_ai_image_script/extractedimages"
    
    # Get all JPG files from all NHTyp folders
    all_images = []
    for folder in ["NHTyp1", "NHTyp2", "NHTyp3", "NHTyp4", "NHTyp5"]:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith('.jpg'):
                    all_images.append(os.path.join(folder_path, file))
    
    if not all_images:
        print("No property images found!")
        return None
    
    # Select a random image
    selected_image = random.choice(all_images)
    print(f"Selected random image: {os.path.basename(selected_image)}")
    print(f"From folder: {os.path.basename(os.path.dirname(selected_image))}")
    return selected_image

def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_property_image_with_gemma(image_path):
    """Send property image to Gemma 3N and get analysis"""
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare the request payload
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create a detailed prompt for property condition assessment
    prompt = """Please analyze this property condition assessment image in detail. I need you to:

1. Describe what you see in the image (rooms, areas, objects, etc.)
2. Assess the overall condition of the property/area shown
3. Identify any visible damage, wear, or maintenance issues
4. Note the quality of construction materials and finishes
5. Comment on cleanliness and general upkeep
6. Point out any safety concerns if visible
7. Provide an overall condition rating (Excellent/Good/Fair/Poor)

Please be thorough and professional in your assessment, as this is for property condition evaluation purposes."""

    payload = {
        "model": "google/gemma-3n-E4B-it",  # Gemma 3N E4B Instruct model
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
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.3
    }
    
    try:
        print(f"Sending property image to Gemma 3N for analysis...")
        print(f"Image: {os.path.basename(image_path)}")
        print("-" * 60)
        
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            ai_response = result['choices'][0]['message']['content']
            print("🏠 PROPERTY CONDITION ASSESSMENT:")
            print("=" * 60)
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
    """Main function"""
    print("🏠 Random Property Condition Assessment Analysis")
    print("=" * 50)
    
    # Get a random property image
    image_path = get_random_property_image()
    
    if not image_path:
        print("❌ No property images found to analyze")
        return
    
    # Analyze the image with Gemma 3N
    response = analyze_property_image_with_gemma(image_path)
    
    if response:
        print("\n" + "=" * 60)
        print("✅ Analysis complete!")
        print(f"📁 Image analyzed: {os.path.basename(image_path)}")
    else:
        print("\n" + "=" * 60)
        print("❌ Failed to get analysis from Gemma 3N")

if __name__ == "__main__":
    main()
