#!/usr/bin/env python3
"""
Test the curated test images (test_images and test_images_batch2 folders)
"""

import requests
import base64
import json
import os
import csv
import re
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from config.env
load_dotenv('config.env')

# Together AI API configuration
API_KEY = os.getenv('TOGETHER_AI_API_KEY')
API_URL = "https://api.together.xyz/v1/chat/completions"

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

def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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
        r'(\d+)\s*-\s*IN-BETWEEN'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    # If no pattern found, return None
    return None

def analyze_property_with_dsm_scoring(image_path):
    """Send property image to Gemma 3N with DSM Neighborhood Scoring System"""
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare the request payload
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # DSM Neighborhood Scoring System Prompt (shortened for batch processing)
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
        print(f"Error analyzing image: {e}")
        return None

def main():
    """Main function"""
    print("🏠 DSM Curated Test Images Analysis")
    print("=" * 50)
    
    # Get curated test images
    images = get_curated_test_images()
    
    if not images:
        print("❌ No curated test images found!")
        return
    
    results = []
    correct_predictions = 0
    
    print(f"Testing {len(images)} curated images...")
    print("-" * 50)
    
    for i, image_info in enumerate(images, 1):
        print(f"Processing {i}/{len(images)}: {image_info['filename']} (Ground Truth: {image_info['ground_truth']})")
        
        # Analyze the image
        response = analyze_property_with_dsm_scoring(image_info['path'])
        
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
            print(f"  ❌ Failed to get response from AI")
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
    csv_filename = f"curated_test_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = f"/home/exouser/DSM-property-condition-assessment/together_ai_image_script/logs/{csv_filename}"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_filename', 'image_path', 'folder', 'ground_truth', 'predicted_score', 'correct', 'ai_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 CURATED TEST IMAGES RESULTS")
    print("=" * 50)
    print(f"Total Images Tested: {total_tests}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Results saved to: {csv_path}")
    
    # Print detailed results
    print("\n📋 DETAILED RESULTS:")
    print("-" * 50)
    for result in results:
        status = "✅" if result['correct'] else "❌"
        print(f"{status} {result['image_filename']} | GT: {result['ground_truth']} | Pred: {result['predicted_score']}")

if __name__ == "__main__":
    main()
