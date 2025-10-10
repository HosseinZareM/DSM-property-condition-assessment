#!/usr/bin/env python3
"""
Copy second batch of test images with renamed format: _class_random_number
"""

import os
import shutil
import random

def copy_second_test_images():
    """Copy the second batch of 10 test images with new naming format"""
    
    # Second batch test images from the accuracy test
    test_images = [
        {
            'original': 'ATT3383_PropertyConditionAssessment_image-20220705-172524.jpg',
            'folder': 'NHTyp3',
            'ground_truth': 3
        },
        {
            'original': 'ATT6954_PropertyConditionAssessment_image-20220801-182647.jpg',
            'folder': 'NHTyp4',
            'ground_truth': 4
        },
        {
            'original': 'ATT36172_PropertyConditionAssessment_image-20220908-195512.jpg',
            'folder': 'NHTyp5',
            'ground_truth': 5
        },
        {
            'original': 'ATT9633_PropertyConditionAssessment_image-20220811-183249.jpg',
            'folder': 'NHTyp2',
            'ground_truth': 2
        },
        {
            'original': 'ATT41979_PropertyConditionAssessment_image-20221005-164438.jpg',
            'folder': 'NHTyp1',
            'ground_truth': 1
        },
        {
            'original': 'ATT6831_PropertyConditionAssessment_image-20220728-194028.jpg',
            'folder': 'NHTyp4',
            'ground_truth': 4
        },
        {
            'original': 'ATT2936_PropertyConditionAssessment_image-20220705-141533.jpg',
            'folder': 'NHTyp3',
            'ground_truth': 3
        },
        {
            'original': 'ATT1588_PropertyConditionAssessment_image-20220628-150626.jpg',
            'folder': 'NHTyp4',
            'ground_truth': 4
        },
        {
            'original': 'ATT280_PropertyConditionAssessment_image-20220622-170404.jpg',
            'folder': 'NHTyp3',
            'ground_truth': 3
        },
        {
            'original': 'ATT891_PropertyConditionAssessment_image-20220624-144314.jpg',
            'folder': 'NHTyp4',
            'ground_truth': 4
        }
    ]
    
    source_base = "/home/exouser/DSM-property-condition-assessment/Data/extractedimages"
    target_dir = "/home/exouser/DSM-property-condition-assessment/test_images_batch2"
    
    # Create new directory for second batch
    os.makedirs(target_dir, exist_ok=True)
    
    print("📁 Copying second batch test images with new naming format...")
    print("=" * 50)
    
    copied_files = []
    
    for i, image_info in enumerate(test_images, 1):
        source_path = os.path.join(source_base, image_info['folder'], image_info['original'])
        
        # Generate random number for each image
        random_num = random.randint(1000, 9999)
        
        # Create new filename: _class_random_number
        new_filename = f"_{image_info['ground_truth']}_{random_num}.jpg"
        target_path = os.path.join(target_dir, new_filename)
        
        try:
            # Copy the file
            shutil.copy2(source_path, target_path)
            
            copied_files.append({
                'original': image_info['original'],
                'new_name': new_filename,
                'class': image_info['ground_truth'],
                'random_num': random_num
            })
            
            print(f"{i:2d}. {image_info['original']}")
            print(f"    → {new_filename} (Class {image_info['ground_truth']})")
            
        except Exception as e:
            print(f"❌ Error copying {image_info['original']}: {e}")
    
    print("\n" + "=" * 50)
    print(f"✅ Successfully copied {len(copied_files)} images to:")
    print(f"📁 {target_dir}")
    
    # Create a summary file
    summary_path = os.path.join(target_dir, "image_mapping_batch2.txt")
    with open(summary_path, 'w') as f:
        f.write("Second Batch Test Images Mapping\n")
        f.write("=" * 40 + "\n\n")
        for item in copied_files:
            f.write(f"Original: {item['original']}\n")
            f.write(f"New Name: {item['new_name']}\n")
            f.write(f"Class: {item['class']}\n")
            f.write(f"Random Number: {item['random_num']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"📄 Summary saved to: {summary_path}")
    
    # Print class distribution
    class_counts = {}
    for item in copied_files:
        class_counts[item['class']] = class_counts.get(item['class'], 0) + 1
    
    print(f"\n📊 Class Distribution:")
    for class_num in sorted(class_counts.keys()):
        print(f"   Class {class_num}: {class_counts[class_num]} images")
    
    return copied_files

if __name__ == "__main__":
    copy_second_test_images()
