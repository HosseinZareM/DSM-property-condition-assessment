#!/usr/bin/env python3
"""
Compare Together AI (Gemma 3N) vs OpenAI (GPT-4 Vision) for DSM Property Assessment
"""

import requests
import base64
import json
import os
import csv
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import io

# Load environment variables from config.env
load_dotenv('config.env')

# API configurations
TOGETHER_API_KEY = os.getenv('TOGETHER_AI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def get_curated_test_images():
    """Get all curated test images from both test folders"""
    test_images = []
    
    # Test images folder
    test_folder1 = "/home/exouser/DSM-property-condition-assessment/test_images"
    if os.path.exists(test_folder1):
        for file in os.listdir(test_folder1):
            if file.lower().endswith('.jpg') and file.startswith('_'):
                parts = file.split('_')
                if len(parts) >= 2:
                    try:
                        class_num = int(parts[1])
                        test_images.append({
                            'path': os.path.join(test_folder1, file),
                            'filename': file,
                            'ground_truth': class_num,
                            'folder': 'test_images'
                        })
                    except ValueError:
                        continue
    
    # Test images batch2 folder
    test_folder2 = "/home/exouser/DSM-property-condition-assessment/test_images_batch2"
    if os.path.exists(test_folder2):
        for file in os.listdir(test_folder2):
            if file.lower().endswith('.jpg') and file.startswith('_'):
                parts = file.split('_')
                if len(parts) >= 2:
                    try:
                        class_num = int(parts[1])
                        test_images.append({
                            'path': os.path.join(test_folder2, file),
                            'filename': file,
                            'ground_truth': class_num,
                            'folder': 'test_images_batch2'
                        })
                    except ValueError:
                        continue
    
    return test_images

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

def extract_score_from_response(response_text):
    """Extract the DSM score from AI response"""
    patterns = [
        r'OVERALL DSM SCORE:\s*(\d+)',
        r'Score:\s*(\d+)',
        r'SCORE:\s*(\d+)',
        r'(\d+)\s*-\s*SLIPPING',
        r'(\d+)\s*-\s*HEALTHY',
        r'(\d+)\s*-\s*UNHEALTHY',
        r'(\d+)\s*-\s*IN-BETWEEN',
        r'(\d+)\s*-\s*VERY HEALTHY',
        r'DSM SCORE:\s*(\d+)',
        r'Property Score:\s*(\d+)',
        r'Assessment Score:\s*(\d+)',
        r'Rating:\s*(\d+)',
        r'(\d+)\s*out of 5',
        r'(\d+)/5'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score
    
    # Look for any number 1-5 in the response as fallback
    numbers = re.findall(r'\b([1-5])\b', response_text)
    if numbers:
        return int(numbers[0])
    
    return None

def analyze_with_together_ai(image_path):
    """Analyze with Together AI Gemma 3N"""
    base64_image = encode_image_to_base64(image_path)
    
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = """You are a professional property assessor using the DSM Neighborhood Scoring System. Analyze this property and provide ONLY the overall DSM score (1-5) and brief justification.

DSM SCORING SYSTEM:
1 = VERY HEALTHY (staying on top of details, all in strong condition)
2 = HEALTHY HOUSE (doing well, small attention to detail missing)
3 = IN-BETWEEN (could go either way, attention to detail missing)
4 = SLIPPING (1-2 red flags, starting to look unhealthy)
5 = UNHEALTHY HOUSE (red flags overwhelming 3+)

ASSESSMENT CRITERIA:
1. General exterior condition
2. Attention to porch/entryway
3. Landscaping
4. Roof, gutters and downspouts
5. Windows
6. Extra personal touches

Provide your response in this exact format:
OVERALL DSM SCORE: [1-5]
JUSTIFICATION: [Brief explanation]"""

    payload = {
        "model": "google/gemma-3n-E4B-it",
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
        "max_tokens": 300,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return None
    except Exception as e:
        print(f"Error with Together AI: {e}")
        return None

def analyze_with_openai(image_path):
    """Analyze with OpenAI GPT-5"""
    base64_image = encode_image_to_base64(image_path)
    
    prompt = f"""You are a professional property assessor using the DSM (Des Moines) Neighborhood Scoring System. Analyze this property condition assessment image and provide ONLY the overall DSM score (1-5) and brief justification.

DSM NEIGHBORHOOD SCORING SYSTEM (1-5 Scale):

SCORE 1 - VERY HEALTHY:
- "Staying on top of the details"
- Roof, porch, windows, yard, landscaping, and details are all in strong condition
- "This house is in great shape; it's very stable"

SCORE 2 - HEALTHY HOUSE:
- "Doing well"
- Small attention to detail missing in roof, porch, windows, yard, landscaping or other areas
- "This house is in good shape"

SCORE 3 - IN-BETWEEN:
- "Could go either way"
- Attention to detail in roof, porch, windows, yard, landscaping or other details missing or not apparent
- "This house is in the middle, in-between."

SCORE 4 - SLIPPING (BEING IGNORED, STARTING TO LOOK UNHEALTHY):
- "1-2 red flags"
- Red flags include: Porch in bad shape, roof in bad condition, landscaping missing or overgrown/neglected, trash, screens torn, etc.
- "This house is slipping."

SCORE 5 - UNHEALTHY HOUSE:
- Red flags overwhelming (3+)
- "This house is clearly unhealthy."

ASSESSMENT CRITERIA TO EVALUATE:
1. General exterior condition - How recently was the house washed or painted?
2. Attention to porch/entryway - What is the condition of the steps, railing, and porch floor?
3. Landscaping - Are the bushes, flowers and lawn well-maintained? Does it seem like the owner is neglecting to do landscaping?
4. Roof, gutters and downspouts - Is the roof warped? Are the gutters clean and well taken care of?
5. Windows - Are the windows, curtains and screen in good condition?
6. Extra personal touches - Are there porch lights, house numbers or thoughtful, seasonally-appropriate decorations? Is this person trying to display effort and pride?

Image: data:image/jpeg;base64,{base64_image}

Provide your response in this exact format:
OVERALL DSM SCORE: [1-5]
JUSTIFICATION: [Brief explanation based on the DSM criteria]"""
    
    try:
        response = openai_client.responses.create(
            model="gpt-5-2025-08-07",
            input=prompt,
            reasoning={
                "effort": "minimal"
            }
        )
        
        # Extract the actual text content from the response
        if response.output and len(response.output) > 1:
            return response.output[1].content[0].text
        else:
            return None
        
    except Exception as e:
        print(f"Error with OpenAI GPT-5: {e}")
        return None

def main():
    """Main function"""
    print("🏠 Model Comparison: Together AI (Gemma 3N) vs OpenAI (GPT-5)")
    print("=" * 80)
    
    # Check API keys
    if not TOGETHER_API_KEY:
        print("❌ Together AI API key not found!")
        return
    if not OPENAI_API_KEY:
        print("❌ OpenAI API key not found!")
        return
    
    # Get test images
    images = get_curated_test_images()
    if not images:
        print("❌ No test images found!")
        return
    
    print(f"Testing {len(images)} images with both models...")
    print("-" * 80)
    
    results = []
    together_correct = 0
    openai_correct = 0
    
    for i, image_info in enumerate(images, 1):
        print(f"Processing {i}/{len(images)}: {image_info['filename']} (GT: {image_info['ground_truth']})")
        
        # Analyze with both models
        together_response = analyze_with_together_ai(image_info['path'])
        openai_response = analyze_with_openai(image_info['path'])
        
        # Extract scores
        together_score = extract_score_from_response(together_response) if together_response else None
        openai_score = extract_score_from_response(openai_response) if openai_response else None
        
        # Check correctness
        together_correct_flag = (together_score == image_info['ground_truth']) if together_score else False
        openai_correct_flag = (openai_score == image_info['ground_truth']) if openai_score else False
        
        if together_correct_flag:
            together_correct += 1
        if openai_correct_flag:
            openai_correct += 1
        
        result = {
            'image_filename': image_info['filename'],
            'ground_truth': image_info['ground_truth'],
            'together_score': together_score,
            'openai_score': openai_score,
            'together_correct': together_correct_flag,
            'openai_correct': openai_correct_flag,
            'together_response': together_response.strip() if together_response else 'No response',
            'openai_response': openai_response.strip() if openai_response else 'No response'
        }
        results.append(result)
        
        print(f"  Together AI: {together_score} ({'✅' if together_correct_flag else '❌'})")
        print(f"  OpenAI: {openai_score} ({'✅' if openai_correct_flag else '❌'})")
    
    # Calculate accuracies
    total = len(results)
    together_accuracy = (together_correct / total) * 100
    openai_accuracy = (openai_correct / total) * 100
    
    # Save results
    csv_filename = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = f"/home/exouser/DSM-property-condition-assessment/together_ai_image_script/logs/{csv_filename}"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_filename', 'ground_truth', 'together_score', 'openai_score', 
                     'together_correct', 'openai_correct', 'together_response', 'openai_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(f"Total Images Tested: {total}")
    print(f"Together AI (Gemma 3N): {together_correct}/{total} ({together_accuracy:.1f}%)")
    print(f"OpenAI (GPT-5): {openai_correct}/{total} ({openai_accuracy:.1f}%)")
    print(f"Results saved to: {csv_path}")
    
    # Determine winner
    if together_accuracy > openai_accuracy:
        print(f"\n🏆 Together AI (Gemma 3N) wins by {together_accuracy - openai_accuracy:.1f}%")
    elif openai_accuracy > together_accuracy:
        print(f"\n🏆 OpenAI (GPT-5) wins by {openai_accuracy - together_accuracy:.1f}%")
    else:
        print(f"\n🤝 It's a tie! Both models achieved {together_accuracy:.1f}% accuracy")

if __name__ == "__main__":
    main()
