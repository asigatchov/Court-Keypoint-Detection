import json
import os
from datetime import datetime

import shutil

def convert_to_coco(json_path, output_dir, limit=-1, orig_width=1280, orig_height=720):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Apply limit if specified
    if limit > 0:
        data = data[:limit]
    
    # Determine source directory (directory where JSON file is located)
    source_dir = os.path.dirname(json_path)
    if not source_dir:
        source_dir = '.'  # Current directory if json_path has no directory part
    
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "tennis_court",
            "supercategory": "court",
            "keypoints": [
                "top_left_outer", "top_right_outer", "bottom_left_outer", "bottom_right_outer",
                "top_left_singles", "bottom_left_singles", "top_right_singles", "bottom_right_singles",
                "top_left_service", "top_right_service",
                "bottom_left_service", "bottom_right_service",
                "net_top_center", "net_bottom_center"
            ],
            "skeleton": [
                [1,2],[2,4],[4,3],[3,1],      # doubles outer
                [5,7],[7,8],[8,6],[6,5],      # singles inner
                [5,9],[7,10],[6,11],[8,12],   # service sides
                [9,10],[11,12],               # service horizontals
                [13,14],                      # net
                [1,5],[2,7],[3,6],[4,8]       # vertical connections
            ]
        }]
    }
    
    ann_id = 1
    for img_id, item in enumerate(data, 1):
        # Extract file name from item id or from the data structure
        if "id" in item:
            file_name = item["id"]
            # Add extension if not present
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                file_name += ".png"
        elif isinstance(item, dict) and "image" in item and "file_name" in item["image"]:
            file_name = item["image"]["file_name"]
        else:
            file_name = f"image_{img_id}.jpg"
        
        coco["images"].append({
            "id": img_id,
            "file_name": file_name,
            "width": orig_width,
            "height": orig_height,
            "date_captured": datetime.now().isoformat()
        })
        
        # Handle different data formats
        if "kps" in item:
            # Original format with kps array
            kps = item["kps"]
        elif isinstance(item, dict) and "annotation" in item and "keypoints" in item["annotation"]:
            # COCO-like format
            kps = []
            keypoints_flat = item["annotation"]["keypoints"]
            for i in range(0, len(keypoints_flat), 3):
                if i + 2 < len(keypoints_flat):
                    x, y, v = keypoints_flat[i], keypoints_flat[i+1], keypoints_flat[i+2]
                    if v > 0:  # visible keypoint
                        kps.append([x, y])
                    else:  # not visible
                        kps.append([-1, -1])  # Use -1 to indicate missing keypoint
        else:
            # Default to empty keypoints
            kps = [[-1, -1]] * 14  # 14 keypoints for tennis court
        keypoints = []
        num_kp = 0
        valid_kps = []
        
        for x, y in kps:
            if x >= 0 and y >= 0:
                keypoints.extend([float(x), float(y), 2])
                valid_kps.append((x, y))
                num_kp += 1
            else:
                keypoints.extend([0.0, 0.0, 0])
        
        if valid_kps:
            xs, ys = zip(*valid_kps)
            x_min, y_min = min(xs), min(ys)
            bbox = [float(x_min), float(y_min), float(max(xs) - x_min), float(max(ys) - y_min)]
            area = bbox[2] * bbox[3]
        else:
            bbox = [0.0, 0.0, float(orig_width), float(orig_height)]
            area = float(orig_width * orig_height)
        
        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "keypoints": keypoints,
            "num_keypoints": num_kp,
            "bbox": [round(v, 1) for v in bbox],  # COCO обычно округляет
            "area": round(area),
            "iscrowd": 0
        })
        ann_id += 1
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save annotations as _annotations.coco.json in the output directory
    output_path = os.path.join(output_dir, "_annotations.coco.json")
    
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=4)
    
    print(f"COCO annotations saved: {output_path}")
    
    # Copy images from source directory to output directory
    copied_count = 0
    for img_info in coco["images"]:
        img_filename = img_info["file_name"]
        source_img_path = os.path.join(source_dir, 'images', img_filename)
        dest_img_path = os.path.join(output_dir, img_filename)
        
        if os.path.exists(source_img_path):
            shutil.copy2(source_img_path, dest_img_path)
            copied_count += 1
        else:
            # Try with images subdirectory as mentioned in requirements
            source_img_path_alt = os.path.join(source_dir, "images", img_filename)
            if os.path.exists(source_img_path_alt):
                shutil.copy2(source_img_path_alt, dest_img_path)
                copied_count += 1
            else:
                print(f"Warning: Image file not found: {source_img_path} or {source_img_path_alt}")
    
    print(f"Copied {copied_count} image files to output directory")

def convert_multiple_datasets(dataset_configs, output_dir, limit=-1, orig_width=1280, orig_height=720):
    """
    Convert multiple datasets (train/val/test) to COCO format
    
    Args:
        dataset_configs: List of tuples (json_path, dataset_type) where dataset_type is 'train', 'val', etc.
        output_dir: Directory to save output files
        limit: Limit number of items to process per dataset (default: -1, no limit)
        orig_width: Original image width
        orig_height: Original image height
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for json_path, dataset_type in dataset_configs:
        print(f"Processing {dataset_type} dataset from {json_path}...")
        convert_to_coco(json_path, output_dir, limit, orig_width, orig_height)
        
        # Rename the output to include dataset type
        input_filename = os.path.splitext(os.path.basename(json_path))[0]
        old_output_path = os.path.join(output_dir, f"{input_filename}_annotations.coco.json")
        new_output_path = os.path.join(output_dir, f"{dataset_type}_{input_filename}_annotations.coco.json")
        
        if os.path.exists(old_output_path):
            os.rename(old_output_path, new_output_path)
            print(f"Renamed output to: {new_output_path}")
        
        old_image_names_path = os.path.join(output_dir, f"{input_filename}_image_names.txt")
        new_image_names_path = os.path.join(output_dir, f"{dataset_type}_{input_filename}_image_names.txt")
        
        if os.path.exists(old_image_names_path):
            os.rename(old_image_names_path, new_image_names_path)
            print(f"Renamed image names file to: {new_image_names_path}")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert tennis court annotations to COCO format")
    parser.add_argument("--json_path", help="Path to input JSON file")
    parser.add_argument("--output_dir", help="Directory to save output files")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of items to process (default: -1, no limit)")
    parser.add_argument("--orig_width", type=int, default=1280, help="Original image width (default: 1280)")
    parser.add_argument("--orig_height", type=int, default=720, help="Original image height (default: 720)")
    
    args = parser.parse_args()
    
    # Process single dataset
    convert_to_coco(args.json_path, args.output_dir, args.limit, args.orig_width, args.orig_height)

# Example: convert_to_coco('data_train.json', 'output_dir', limit=100)