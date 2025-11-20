#!/usr/bin/env python3
"""
Compare Zero-Shot vs Few-Shot Learning for DSM Property Assessment
"""

import requests
import base64
import json
import os
import random
import csv
import re
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from config.env
load_dotenv('config.env')

# Together AI API configuration
API_KEY = os.getenv('TOGETHER_AI_API_KEY')
API_URL = "https://api.together.xyz/v1/chat/completions"

def get_example_images():
    """Get example images for each DSM score category (1-5)"""
    base_path = "/home/exouser/DSM-property-condition-assessment/Data/extractedimages"
    
    examples = {}
    
    # Get one example image from each NHTyp folder
    for folder in ["NHTyp1", "NHTyp2", "NHTyp3", "NHTyp4", "NHTyp5"]:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
            if jpg_files:
                # Take the first image as example
                example_file = jpg_files[0]
                score = int(folder.replace('NHTyp', ''))
                examples[score] = {
                    'path': os.path.join(folder_path, example_file),
                    'filename': example_file,
                    'score': score
                }
    
    return examples

def get_test_images(count=10):
    """Get random property images from all folders, excluding example images"""
    base_path = "/home/exouser/DSM-property-condition-assessment/Data/extractedimages"
    
    # Get example images to exclude them from test set
    example_images = get_example_images()
    example_filenames = {ex['filename'] for ex in example_images.values()}
    
    # Get all JPG files from all NHTyp folders, excluding examples
    all_images = []
    for folder in ["NHTyp1", "NHTyp2", "NHTyp3", "NHTyp4", "NHTyp5"]:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith('.jpg') and file not in example_filenames:
                    all_images.append({
                        'path': os.path.join(folder_path, file),
                        'filename': file,
                        'folder': folder,
                        'ground_truth': int(folder.replace('NHTyp', ''))
                    })
    
    if len(all_images) < count:
        print(f"Only {len(all_images)} images available (excluding examples), using all of them")
        return all_images
    
    return random.sample(all_images, count)

def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_score_from_response(response_text):
    """Extract the DSM score from AI response"""
    patterns = [
        r'OVERALL DSM SCORE:\s*(\d+)',
        r'Score:\s*(\d+)',
        r'SCORE:\s*(\d+)',
        r'(\d+)\s*-\s*SLIPPING',
        r'(\d+)\s*-\s*HEALTHY',
        r'(\d+)\s*-\s*UNHEALTHY',
        r'(\d+)\s*-\s*IN-BETWEEN'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return None

def analyze_property_zero_shot(image_path):
    """Zero-shot analysis using original prompt"""
    base64_image = encode_image_to_base64(image_path)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
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
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return None
            
    except Exception as e:
        print(f"Error in zero-shot analysis: {e}")
        return None

def analyze_property_few_shot(image_path, example_images):
    """Few-shot analysis using example images"""
    base64_image = encode_image_to_base64(image_path)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Build few-shot examples content
    examples_content = []
    
    examples_content.append({
        "type": "text",
        "text": """You are a professional property assessor using the DSM Neighborhood Scoring System. I will show you examples of each score level, then ask you to score a new property.

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

Here are examples of each score level:

"""
    })
    
    # Add example images for each score (1-5)
    for score in range(1, 6):
        if score in example_images:
            example_path = example_images[score]['path']
            example_base64 = encode_image_to_base64(example_path)
            
            examples_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{example_base64}"
                }
            })
            
            score_descriptions = {
                1: "This is an example of DSM Score 1 - VERY HEALTHY: The property shows excellent maintenance with all details attended to, pristine condition throughout.",
                2: "This is an example of DSM Score 2 - HEALTHY HOUSE: The property is well-maintained with only minor attention to detail missing.",
                3: "This is an example of DSM Score 3 - IN-BETWEEN: The property could go either way, with some attention to detail missing but not critical issues.",
                4: "This is an example of DSM Score 4 - SLIPPING: The property shows 1-2 red flags and is starting to look unhealthy with noticeable maintenance issues.",
                5: "This is an example of DSM Score 5 - UNHEALTHY HOUSE: The property has overwhelming red flags (3+) with significant maintenance problems."
            }
            
            examples_content.append({
                "type": "text",
                "text": f"{score_descriptions[score]}\n"
            })
    
    examples_content.append({
        "type": "text",
        "text": """Now, please analyze this new property image and provide your assessment in this exact format:

OVERALL DSM SCORE: [1-5]
JUSTIFICATION: [Brief explanation based on the examples above]"""
    })
    
    examples_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    })
    
    payload = {
        "model": "google/gemma-3n-E4B-it",
        "messages": [
            {
                "role": "user",
                "content": examples_content
            }
        ],
        "max_tokens": 400,
        "temperature": 0.2
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return None
            
    except Exception as e:
        print(f"Error in few-shot analysis: {e}")
        return None

def main():
    """Main function to compare zero-shot vs few-shot learning"""
    print("🏠 DSM Zero-Shot vs Few-Shot Learning Comparison")
    print("=" * 60)
    
    # Get example images for few-shot learning
    print("Loading example images for few-shot learning...")
    example_images = get_example_images()
    
    if len(example_images) < 5:
        print(f"❌ Only found {len(example_images)} example images. Need examples for all 5 score categories!")
        return
    
    print(f"✅ Found examples for scores: {sorted(example_images.keys())}")
    
    # Get test images
    print("Loading test images...")
    images = get_test_images(10)
    
    if not images:
        print("❌ No test images found!")
        return
    
    print(f"Testing {len(images)} images with both approaches...")
    print("-" * 60)
    
    results = []
    zero_shot_correct = 0
    few_shot_correct = 0
    
    for i, image_info in enumerate(images, 1):
        print(f"\nProcessing {i}/{len(images)}: {image_info['filename']} (Ground Truth: {image_info['ground_truth']})")
        
        # Zero-shot analysis
        print("  Running zero-shot analysis...")
        zero_shot_response = analyze_property_zero_shot(image_info['path'])
        zero_shot_score = extract_score_from_response(zero_shot_response) if zero_shot_response else None
        
        # Few-shot analysis
        print("  Running few-shot analysis...")
        few_shot_response = analyze_property_few_shot(image_info['path'], example_images)
        few_shot_score = extract_score_from_response(few_shot_response) if few_shot_response else None
        
        # Check correctness
        zero_shot_correct_flag = (zero_shot_score == image_info['ground_truth']) if zero_shot_score else False
        few_shot_correct_flag = (few_shot_score == image_info['ground_truth']) if few_shot_score else False
        
        if zero_shot_correct_flag:
            zero_shot_correct += 1
        if few_shot_correct_flag:
            few_shot_correct += 1
        
        result = {
            'image_filename': image_info['filename'],
            'image_path': image_info['path'],
            'folder': image_info['folder'],
            'ground_truth': image_info['ground_truth'],
            'zero_shot_score': zero_shot_score,
            'zero_shot_correct': zero_shot_correct_flag,
            'zero_shot_response': zero_shot_response.strip() if zero_shot_response else 'No response',
            'few_shot_score': few_shot_score,
            'few_shot_correct': few_shot_correct_flag,
            'few_shot_response': few_shot_response.strip() if few_shot_response else 'No response'
        }
        results.append(result)
        
        print(f"  Zero-shot: {zero_shot_score} ({'✅' if zero_shot_correct_flag else '❌'})")
        print(f"  Few-shot:  {few_shot_score} ({'✅' if few_shot_correct_flag else '❌'})")
    
    # Calculate accuracies
    total_tests = len(results)
    zero_shot_accuracy = (zero_shot_correct / total_tests) * 100 if total_tests > 0 else 0
    few_shot_accuracy = (few_shot_correct / total_tests) * 100 if total_tests > 0 else 0
    
    # Save results to CSV
    csv_filename = f"zero_vs_few_shot_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = f"/home/exouser/DSM-property-condition-assessment/together_ai_image_script/logs/{csv_filename}"
    
    # Ensure logs directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_filename', 'image_path', 'folder', 'ground_truth', 
                     'zero_shot_score', 'zero_shot_correct', 'zero_shot_response',
                     'few_shot_score', 'few_shot_correct', 'few_shot_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS")
    print("=" * 60)
    print(f"Total Images Tested: {total_tests}")
    print(f"Zero-Shot Correct: {zero_shot_correct} | Accuracy: {zero_shot_accuracy:.1f}%")
    print(f"Few-Shot Correct:  {few_shot_correct} | Accuracy: {few_shot_accuracy:.1f}%")
    print(f"Improvement: {few_shot_accuracy - zero_shot_accuracy:+.1f} percentage points")
    print(f"Results saved to: {csv_path}")
    
    # Print detailed results
    print("\n📋 DETAILED RESULTS:")
    print("-" * 60)
    print(f"{'Image':<20} {'GT':<2} {'Zero':<2} {'Few':<2} {'Zero':<5} {'Few':<5}")
    print("-" * 60)
    for result in results:
        zero_status = "✅" if result['zero_shot_correct'] else "❌"
        few_status = "✅" if result['few_shot_correct'] else "❌"
        print(f"{result['image_filename']:<20} {result['ground_truth']:<2} {result['zero_shot_score'] or 'E':<2} {result['few_shot_score'] or 'E':<2} {zero_status:<5} {few_status:<5}")
    
    # Print example images used
    print("\n📸 EXAMPLE IMAGES USED FOR FEW-SHOT LEARNING:")
    print("-" * 60)
    for score in sorted(example_images.keys()):
        print(f"Score {score}: {example_images[score]['filename']}")

if __name__ == "__main__":
    main()

