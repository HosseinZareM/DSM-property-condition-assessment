#!/usr/bin/env python3
"""
Test Few-Shot Learning Pipeline with GPT on Test-images Directory
"""

import requests
import base64
import json
import os
import re
import csv
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from config.env
load_dotenv('config.env')

# OpenAI API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

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

def get_test_images():
    """Get all images from Test-images directory with their ground truth scores"""
    test_path = "/home/exouser/DSM-property-condition-assessment/Data/Test-images"
    
    test_images = []
    
    if not os.path.exists(test_path):
        print(f"❌ Test-images directory not found: {test_path}")
        return []
    
    for filename in os.listdir(test_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Extract ground truth score from filename (e.g., "1_Picture3.png" -> score 1)
            try:
                ground_truth = int(filename.split('_')[0])
                test_images.append({
                    'path': os.path.join(test_path, filename),
                    'filename': filename,
                    'ground_truth': ground_truth
                })
            except (ValueError, IndexError):
                print(f"⚠️  Could not extract ground truth from filename: {filename}")
                continue
    
    return sorted(test_images, key=lambda x: x['ground_truth'])

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

def analyze_property_gpt_few_shot(image_path, example_images):
    """GPT few-shot analysis using example images"""
    base64_image = encode_image_to_base64(image_path)
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Build few-shot examples content for GPT
    examples_content = []
    
    # Add text explanation of the scoring system
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
            
            # Add example image
            examples_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{example_base64}"
                }
            })
            
            # Add score explanation for this example
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
    
    # Add instruction for the new image
    examples_content.append({
        "type": "text",
        "text": """Now, please analyze this new property image and provide your assessment in this exact format:

OVERALL DSM SCORE: [1-5]
JUSTIFICATION: [Brief explanation based on the examples above]"""
    })
    
    # Add the test image
    examples_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    })
    
    payload = {
        "model": "gpt-4o",  # Using GPT-4 Vision
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
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return None
            
    except Exception as e:
        print(f"Error analyzing image with GPT: {e}")
        return None

def main():
    """Main function to test GPT few-shot pipeline on Test-images"""
    print("🏠 GPT Few-Shot Learning Pipeline Test - Test-images Directory")
    print("=" * 70)
    
    # Get example images for few-shot learning
    print("Loading example images for each DSM score...")
    example_images = get_example_images()
    
    if len(example_images) < 5:
        print(f"❌ Only found {len(example_images)} example images. Need examples for all 5 score categories!")
        return
    
    print(f"✅ Found examples for scores: {sorted(example_images.keys())}")
    print("\n📸 Example images being used:")
    for score in sorted(example_images.keys()):
        print(f"  Score {score}: {example_images[score]['filename']}")
    
    # Get test images from Test-images directory
    print("\nLoading test images from Test-images directory...")
    test_images = get_test_images()
    
    if not test_images:
        print("❌ No test images found in Test-images directory!")
        return
    
    print(f"✅ Found {len(test_images)} test images:")
    for img in test_images:
        print(f"  {img['filename']} (Ground Truth: {img['ground_truth']})")
    
    print(f"\nTesting {len(test_images)} images with GPT few-shot learning...")
    print("-" * 70)
    
    results = []
    correct_predictions = 0
    
    for i, image_info in enumerate(test_images, 1):
        print(f"\n🔍 Processing {i}/{len(test_images)}: {image_info['filename']}")
        print(f"   Ground Truth: {image_info['ground_truth']}")
        
        # Analyze the image with GPT few-shot learning
        response = analyze_property_gpt_few_shot(image_info['path'], example_images)
        
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
                    'ground_truth': image_info['ground_truth'],
                    'predicted_score': predicted_score,
                    'correct': is_correct,
                    'ai_response': response.strip()
                }
                results.append(result)
                
                print(f"   Predicted Score: {predicted_score}")
                print(f"   Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
                print(f"   AI Response: {response.strip()}")
            else:
                print(f"   ❌ Could not extract score from response")
                print(f"   AI Response: {response.strip()}")
                result = {
                    'image_filename': image_info['filename'],
                    'image_path': image_info['path'],
                    'ground_truth': image_info['ground_truth'],
                    'predicted_score': 'ERROR',
                    'correct': False,
                    'ai_response': response.strip()
                }
                results.append(result)
        else:
            print(f"   ❌ Failed to get response from GPT")
            result = {
                'image_filename': image_info['filename'],
                'image_path': image_info['path'],
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
    csv_filename = f"gpt_few_shot_test_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = f"/home/exouser/DSM-property-condition-assessment/together_ai_image_script/logs/{csv_filename}"
    
    # Ensure logs directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_filename', 'image_path', 'ground_truth', 'predicted_score', 'correct', 'ai_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 GPT FEW-SHOT LEARNING TEST RESULTS")
    print("=" * 70)
    print(f"Total Images Tested: {total_tests}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Results saved to: {csv_path}")
    
    # Print detailed results
    print("\n📋 DETAILED RESULTS:")
    print("-" * 70)
    print(f"{'Image':<25} {'GT':<2} {'Pred':<4} {'Result':<8}")
    print("-" * 70)
    for result in results:
        status = "✅" if result['correct'] else "❌"
        pred = result['predicted_score'] if result['predicted_score'] != 'ERROR' else 'E'
        print(f"{result['image_filename']:<25} {result['ground_truth']:<2} {pred:<4} {status:<8}")
    
    # Show score distribution
    print("\n📊 SCORE DISTRIBUTION:")
    print("-" * 70)
    ground_truth_counts = {}
    predicted_counts = {}
    
    for result in results:
        gt = result['ground_truth']
        pred = result['predicted_score'] if result['predicted_score'] != 'ERROR' else 'ERROR'
        
        ground_truth_counts[gt] = ground_truth_counts.get(gt, 0) + 1
        predicted_counts[pred] = predicted_counts.get(pred, 0) + 1
    
    print("Ground Truth Distribution:")
    for score in sorted(ground_truth_counts.keys()):
        print(f"  Score {score}: {ground_truth_counts[score]} images")
    
    print("\nPredicted Distribution:")
    for score in sorted(predicted_counts.keys()):
        print(f"  Score {score}: {predicted_counts[score]} images")

if __name__ == "__main__":
    main()

