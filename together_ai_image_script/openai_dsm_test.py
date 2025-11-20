#!/usr/bin/env python3
"""
OpenAI GPT-4 Vision DSM Property Condition Assessment Test
"""

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

# OpenAI API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

def get_curated_test_images():
    """Get all curated test images from both test folders"""
    test_images = []
    
    # Test images folder
    test_folder1 = "/home/exouser/DSM-property-condition-assessment/test_images"
    if os.path.exists(test_folder1):
        for file in os.listdir(test_folder1):
            if file.lower().endswith('.jpg') and file.startswith('_'):
                # Extract class from filename (e.g., _2_4916.jpg -> class 2)
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
                # Extract class from filename (e.g., _2_4916.jpg -> class 2)
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
    # Look for patterns like "Score: 4" or "OVERALL DSM SCORE: 3"
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
            if 1 <= score <= 5:  # Validate score is in valid range
                return score
    
    # Look for any number 1-5 in the response as fallback
    numbers = re.findall(r'\b([1-5])\b', response_text)
    if numbers:
        return int(numbers[0])
    
    # If no pattern found, return None
    return None

def analyze_property_with_openai_gpt5(image_path):
    """Send property image to OpenAI GPT-5 with DSM Neighborhood Scoring System"""
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # DSM Neighborhood Scoring System Prompt for OpenAI GPT-5
    prompt = f"""You are a professional property assessor using the DSM (Des Moines) Neighborhood Scoring System. Analyze this property condition assessment image and provide ONLY the overall DSM score (1-5) and brief justification.

DSM NEIGHBORHOOD SCORING SYSTEM (1-5 Scale):

SCORE 1 - VERY HEALTHY:
- "Staying on top of the details"
- Roof, porch, windows, yard, landscaping, and details are all in strong condition
- "This house is in great shape; it's very stable"
- "I am impressed by this house. They would have no problem finding a buyer for this house soon."

SCORE 2 - HEALTHY HOUSE:
- "Doing well"
- Small attention to detail missing in roof, porch, windows, yard, landscaping or other areas
- "This house is in good shape"
- "It might just need a few small changes or updates to fill in the details (landscaping, porch)."

SCORE 3 - IN-BETWEEN:
- "Could go either way"
- Attention to detail in roof, porch, windows, yard, landscaping or other details missing or not apparent
- "This house is in the middle, in-between."
- "I can't tell if someone is actively investing in this house… I don't see red flags but I also don't see obvious recent effort."

SCORE 4 - SLIPPING (BEING IGNORED, STARTING TO LOOK UNHEALTHY):
- "1-2 red flags"
- Red flags include: Porch in bad shape, roof in bad condition, landscaping missing or overgrown/neglected, trash, screens torn, etc.
- "This house is slipping."
- "This house is starting to experience some neglect. I see at least 1-2 red flags of distress here."

SCORE 5 - UNHEALTHY HOUSE:
- Red flags overwhelming (3+)
- "This house is clearly unhealthy."
- "I see a few red flags here… no one has invested in this house for a while."

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
        response = client.responses.create(
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
        print(f"Error analyzing image with OpenAI GPT-5: {e}")
        return None

def main():
    """Main function"""
    print("🏠 OpenAI GPT-5 DSM Property Assessment Test")
    print("=" * 60)
    
    # Check if API key is available
    if not OPENAI_API_KEY:
        print("❌ OpenAI API key not found in config.env!")
        print("Please add OPENAI_API_KEY=your_key_here to config.env")
        return
    
    # Get curated test images
    images = get_curated_test_images()
    
    if not images:
        print("❌ No curated test images found!")
        return
    
    results = []
    correct_predictions = 0
    
    print(f"Testing {len(images)} curated images with OpenAI GPT-5...")
    print("-" * 60)
    
    for i, image_info in enumerate(images, 1):
        print(f"Processing {i}/{len(images)}: {image_info['filename']} (Ground Truth: {image_info['ground_truth']})")
        
        # Analyze the image
        response = analyze_property_with_openai_gpt5(image_info['path'])
        
        if response:
            # Extract score from response
            predicted_score = extract_score_from_response(response)
            
            if predicted_score:
                is_correct = (predicted_score == image_info['ground_truth'])
                if is_correct:
                    correct_predictions += 1
                
                result = {
                    'image_filename': image_info['filename'],
                    'image_path': image_info['path'],
                    'folder': image_info['folder'],
                    'ground_truth': image_info['ground_truth'],
                    'predicted_score': predicted_score,
                    'correct': is_correct,
                    'ai_response': response.strip()
                }
                results.append(result)
                
                print(f"  Ground Truth: {image_info['ground_truth']}, Predicted: {predicted_score}, Correct: {is_correct}")
            else:
                print(f"  ❌ Could not extract score from response")
                result = {
                    'image_filename': image_info['filename'],
                    'image_path': image_info['path'],
                    'folder': image_info['folder'],
                    'ground_truth': image_info['ground_truth'],
                    'predicted_score': 'ERROR',
                    'correct': False,
                    'ai_response': response.strip()
                }
                results.append(result)
        else:
            print(f"  ❌ Failed to get response from OpenAI")
            result = {
                'image_filename': image_info['filename'],
                'image_path': image_info['path'],
                'folder': image_info['folder'],
                'ground_truth': image_info['ground_truth'],
                'predicted_score': 'ERROR',
                'correct': False,
                'ai_response': 'No response'
            }
            results.append(result)
    
    # Calculate accuracy
    total_tests = len(results)
    accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
    
    # Save results to CSV
    csv_filename = f"openai_gpt4v_dsm_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = f"/home/exouser/DSM-property-condition-assessment/together_ai_image_script/logs/{csv_filename}"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_filename', 'image_path', 'folder', 'ground_truth', 'predicted_score', 'correct', 'ai_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 OPENAI GPT-5 DSM TEST RESULTS")
    print("=" * 60)
    print(f"Total Images Tested: {total_tests}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Results saved to: {csv_path}")
    
    # Print detailed results
    print("\n📋 DETAILED RESULTS:")
    print("-" * 60)
    for result in results:
        status = "✅" if result['correct'] else "❌"
        print(f"{status} {result['image_filename']} | GT: {result['ground_truth']} | Pred: {result['predicted_score']}")
    
    # Class-wise accuracy breakdown
    print("\n📈 CLASS-WISE ACCURACY BREAKDOWN:")
    print("-" * 60)
    class_stats = {}
    for result in results:
        gt = result['ground_truth']
        if gt not in class_stats:
            class_stats[gt] = {'total': 0, 'correct': 0}
        class_stats[gt]['total'] += 1
        if result['correct']:
            class_stats[gt]['correct'] += 1
    
    for class_num in sorted(class_stats.keys()):
        stats = class_stats[class_num]
        accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"Class {class_num}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")

if __name__ == "__main__":
    main()
