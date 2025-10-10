#!/usr/bin/env python3
"""
Copy test images with renamed format: _class_random_number
"""

import os
import shutil
import random

def copy_test_images():
    """Copy the 10 test images with new naming format"""
    
    # Test images from the accuracy test
    test_images = [
        {
            'original': 'ATT13833_PropertyConditionAssessment_image-20220902-141043.jpg',
            'folder': 'NHTyp5',
            'ground_truth': 5
        },
        {
            'original': 'ATT903_PropertyConditionAssessment_image-20220624-145148.jpg',
            'folder': 'NHTyp4',
            'ground_truth': 4
        },
        {
            'original': 'ATT9784_PropertyConditionAssessment_image-20220811-195442.jpg',
            'folder': 'NHTyp2',
            'ground_truth': 2
        },
        {
            'original': 'ATT8404_PropertyConditionAssessment_image-20220809-181906.jpg',
            'folder': 'NHTyp2',
            'ground_truth': 2
        },
        {
            'original': 'ATT3873_PropertyConditionAssessment_image-20220706-185058.jpg',
            'folder': 'NHTyp3',
            'ground_truth': 3
        },
        {
            'original': 'ATT2592_PropertyConditionAssessment_image-20220630-185516.jpg',
            'folder': 'NHTyp3',
            'ground_truth': 3
        },
        {
            'original': 'ATT8988_PropertyConditionAssessment_image-20220810-175227.jpg',
            'folder': 'NHTyp2',
            'ground_truth': 2
        },
        {
            'original': 'ATT13618_PropertyConditionAssessment_image-20220901-181413.jpg',
            'folder': 'NHTyp5',
            'ground_truth': 5
        },
        {
            'original': 'ATT9154_PropertyConditionAssessment_image-20220811-145606.jpg',
            'folder': 'NHTyp2',
            'ground_truth': 2
        },
        {
            'original': 'ATT9634_PropertyConditionAssessment_image-20220811-183214.jpg',
            'folder': 'NHTyp2',
            'ground_truth': 2
        }
    ]
    
    source_base = "/home/exouser/DSM-property-condition-assessment/Data/extractedimages"
    target_dir = "/home/exouser/DSM-property-condition-assessment/test_images"
    
    print("📁 Copying test images with new naming format...")
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
    summary_path = os.path.join(target_dir, "image_mapping.txt")
    with open(summary_path, 'w') as f:
        f.write("Test Images Mapping\n")
        f.write("=" * 30 + "\n\n")
        for item in copied_files:
            f.write(f"Original: {item['original']}\n")
            f.write(f"New Name: {item['new_name']}\n")
            f.write(f"Class: {item['class']}\n")
            f.write(f"Random Number: {item['random_num']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"📄 Summary saved to: {summary_path}")
    
    return copied_files

if __name__ == "__main__":
    copy_test_images()
