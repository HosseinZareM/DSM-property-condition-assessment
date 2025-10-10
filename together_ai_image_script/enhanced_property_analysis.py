#!/usr/bin/env python3
"""
Enhanced Property Condition Assessment with DSM Neighborhood Scoring System
"""

import requests
import base64
import json
import os
import random
from PIL import Image
import io

# Together AI API configuration
API_KEY = "tgp_v1_20wpGwgQcqOZn5aaoYA_-NihgYGHUYks7i44R9AecfQ"
API_URL = "https://api.together.xyz/v1/chat/completions"

def get_random_property_image():
    """Get a random property condition assessment image"""
    base_path = "/home/exouser/DSM-property-condition-assessment/Data/extractedimages"
    
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

def analyze_property_with_dsm_scoring(image_path):
    """Send property image to Gemma 3N with DSM Neighborhood Scoring System"""
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare the request payload
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # DSM Neighborhood Scoring System Prompt
    prompt = """You are a professional property assessor using the DSM (Des Moines) Neighborhood Scoring System. Please analyze this property condition assessment image and provide a comprehensive evaluation.

## DSM NEIGHBORHOOD SCORING SYSTEM (1-5 Scale):

**SCORE 1 - VERY HEALTHY:**
- "Staying on top of the details"
- Roof, porch, windows, yard, landscaping, and details are all in strong condition
- "This house is in great shape; it's very stable"
- "I am impressed by this house. They would have no problem finding a buyer for this house soon."

**SCORE 2 - HEALTHY HOUSE:**
- "Doing well"
- Small attention to detail missing in roof, porch, windows, yard, landscaping or other areas
- "This house is in good shape"
- "It might just need a few small changes or updates to fill in the details (landscaping, porch)."

**SCORE 3 - IN-BETWEEN:**
- "Could go either way"
- Attention to detail in roof, porch, windows, yard, landscaping or other details missing or not apparent
- "This house is in the middle, in-between."
- "I can't tell if someone is actively investing in this house… I don't see red flags but I also don't see obvious recent effort."

**SCORE 4 - SLIPPING (BEING IGNORED, STARTING TO LOOK UNHEALTHY):**
- "1-2 red flags"
- Red flags include: Porch in bad shape, roof in bad condition, landscaping missing or overgrown/neglected, trash, screens torn, etc.
- "This house is slipping."
- "This house is starting to experience some neglect. I see at least 1-2 red flags of distress here."

**SCORE 5 - UNHEALTHY HOUSE:**
- Red flags overwhelming (3+)
- "This house is clearly unhealthy."
- "I see a few red flags here… no one has invested in this house for a while."

## ASSESSMENT CRITERIA TO EVALUATE:

1. **General exterior condition**
   - How recently was the house washed or painted?

2. **Attention to porch/entryway**
   - What is the condition of the steps, railing, and porch floor?

3. **Landscaping**
   - Are the bushes, flowers and lawn well-maintained?
   - Does it seem like the owner is neglecting to do landscaping?

4. **Roof, gutters and downspouts**
   - Is the roof warped?
   - Are the gutters clean and well taken care of?

5. **Windows**
   - Are the windows, curtains and screen in good condition?

6. **Extra personal touches**
   - Are there porch lights, house numbers or thoughtful, seasonally-appropriate decorations?
   - Is this person trying to display effort and pride?

## YOUR TASK:

Please provide a detailed analysis following this format:

### PROPERTY DESCRIPTION:
[Describe what you see in the image - structure, condition, surroundings]

### DETAILED ASSESSMENT BY CRITERIA:
1. **General exterior condition**: [Your assessment]
2. **Porch/entryway**: [Your assessment]
3. **Landscaping**: [Your assessment]
4. **Roof, gutters and downspouts**: [Your assessment]
5. **Windows**: [Your assessment]
6. **Extra personal touches**: [Your assessment]

### RED FLAGS IDENTIFIED:
[List any red flags you observe]

### OVERALL DSM SCORE: [1-5]
**Score: [NUMBER] - [SCORE DESCRIPTION]**

### JUSTIFICATION:
[Explain why you gave this score based on the DSM criteria]

### RECOMMENDATIONS:
[Suggest specific improvements to raise the score]

Please be thorough and professional in your assessment using the DSM Neighborhood Scoring System."""

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
        "max_tokens": 1500,
        "temperature": 0.3
    }
    
    try:
        print(f"Sending property image to Gemma 3N for DSM Neighborhood Scoring...")
        print(f"Image: {os.path.basename(image_path)}")
        print("-" * 70)
        
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            ai_response = result['choices'][0]['message']['content']
            print("🏠 DSM NEIGHBORHOOD SCORING ASSESSMENT:")
            print("=" * 70)
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
    print("🏠 DSM Neighborhood Scoring Property Assessment")
    print("=" * 50)
    
    # Get a random property image
    image_path = get_random_property_image()
    
    if not image_path:
        print("❌ No property images found to analyze")
        return
    
    # Analyze the image with DSM scoring system
    response = analyze_property_with_dsm_scoring(image_path)
    
    if response:
        print("\n" + "=" * 70)
        print("✅ DSM Assessment complete!")
        print(f"📁 Image analyzed: {os.path.basename(image_path)}")
    else:
        print("\n" + "=" * 70)
        print("❌ Failed to get DSM assessment from Gemma 3N")

if __name__ == "__main__":
    main()
