import os
import shutil
import argparse
import random
from pathlib import Path


def get_file_pairs(input_dir):
    """
    Find all JPG/JSON file pairs in the input directory.
    Returns a list of tuples (jpg_path, json_path).
    """
    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
    file_pairs = []
    
    for jpg_file in jpg_files:
        base_name = os.path.splitext(jpg_file)[0]
        json_file = base_name + '.json'
        
        jpg_path = os.path.join(input_dir, jpg_file)
        json_path = os.path.join(input_dir, json_file)
        
        # Check if corresponding JSON file exists
        if os.path.exists(json_path):
            file_pairs.append((jpg_path, json_path))
        else:
            print(f"Warning: Found JPG file without corresponding JSON: {jpg_file}")
    
    return file_pairs


def split_dataset(input_dir, output_dir, train_ratio=0.8):
    """
    Split the dataset into train and validation sets.
    """
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    
    # Get all file pairs
    file_pairs = get_file_pairs(input_dir)
    
    if not file_pairs:
        print("No JPG/JSON file pairs found in the input directory.")
        return
    
    # Shuffle the file pairs
    random.shuffle(file_pairs)
    
    # Calculate split index
    split_idx = int(len(file_pairs) * train_ratio)
    
    # Split into train and validation
    train_pairs = file_pairs[:split_idx]
    valid_pairs = file_pairs[split_idx:]
    
    print(f"Total pairs: {len(file_pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(valid_pairs)}")
    
    # Copy train files
    for jpg_path, json_path in train_pairs:
        jpg_filename = os.path.basename(jpg_path)
        json_filename = os.path.basename(json_path)
        
        # Copy JPG file
        dst_jpg_path = os.path.join(train_dir, jpg_filename)
        shutil.copy2(jpg_path, dst_jpg_path)
        
        # Copy JSON file
        dst_json_path = os.path.join(train_dir, json_filename)
        shutil.copy2(json_path, dst_json_path)
    
    # Copy validation files
    for jpg_path, json_path in valid_pairs:
        jpg_filename = os.path.basename(jpg_path)
        json_filename = os.path.basename(json_path)
        
        # Copy JPG file
        dst_jpg_path = os.path.join(valid_dir, jpg_filename)
        shutil.copy2(jpg_path, dst_jpg_path)
        
        # Copy JSON file
        dst_json_path = os.path.join(valid_dir, json_filename)
        shutil.copy2(json_path, dst_json_path)
    
    print(f"Dataset split completed!")
    print(f"Train files copied to: {train_dir}")
    print(f"Validation files copied to: {valid_dir}")


def main():
    parser = argparse.ArgumentParser(description='Split JPG/JSON dataset into train and validation sets.')
    parser.add_argument('--input_dir', required=True, help='Input directory containing JPG/JSON file pairs')
    parser.add_argument('--output_dir', required=True, help='Output directory to create train/valid subdirectories')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split the dataset
    split_dataset(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()