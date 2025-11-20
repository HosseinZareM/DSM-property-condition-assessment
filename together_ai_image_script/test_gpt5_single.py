#!/usr/bin/env python3
"""
Test OpenAI GPT-5 with a single image
"""

import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import io

# Load environment variables from config.env
load_dotenv('config.env')

# OpenAI API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

def encode_image_to_base64(image_path, max_size=512):
    """Convert image file to base64 string with size optimization"""
    # Open and resize image to reduce context window usage
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image to max_size while maintaining aspect ratio
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Save to bytes with compression
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
        img_byte_arr = img_byte_arr.getvalue()
        
        return base64.b64encode(img_byte_arr).decode('utf-8')

def test_gpt5_api():
    """Test OpenAI GPT-5 API with a single image"""
    
    # Use the first test image
    image_path = "/home/exouser/DSM-property-condition-assessment/test_images/_3_3665.jpg"
    
    if not os.path.exists(image_path):
        print("❌ Test image not found!")
        return
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    prompt = f"""Analyze this property image and provide a DSM score from 1-5 where:
1 = Very Healthy
2 = Healthy House  
3 = In-Between
4 = Slipping
5 = Unhealthy House

Image: data:image/jpeg;base64,{base64_image}

Provide your response as: SCORE: [1-5]"""
    
    try:
        print("Testing OpenAI GPT-5 API...")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Model: gpt-5-2025-08-07")
        print("-" * 50)
        
        response = client.responses.create(
            model="gpt-5-2025-08-07",
            input=prompt,
            reasoning={
                "effort": "minimal"
            }
        )
        
        print("✅ Success! OpenAI GPT-5 Response:")
        # Extract the actual text content from the response
        if response.output and len(response.output) > 1:
            content = response.output[1].content[0].text
            print(content)
        else:
            print("No output content found")
        
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_gpt5_api()
