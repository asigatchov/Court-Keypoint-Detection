#!/usr/bin/env python3
"""
Volleyball court keypoints annotation tool
Allows manual annotation of volleyball court keypoints on images
"""
import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

class VolleyballCourtAnnotator:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.image_files = []
        self.current_image_idx = 0
        self.current_image = None
        self.current_annotations = None
        
        # Volleyball court keypoints (8 points)
        self.keypoints_names = [
            "1_back_left",      # 0
            "2_back_left",      # 1
            "3_back_right",     # 2
            "4_back_right",     # 3
            "5_center_left",    # 4
            "6_center_right",   # 5
            "7_net_left",       # 6
            "8_net_right"       # 7
        ]
        
        # Keypoint colors (BGR format) - distinct and contrasting
        self.keypoint_colors = [
            (0, 0, 255),    # 0 - Red
            (0, 165, 255),  # 1 - Orange
            (0, 255, 255),  # 2 - Yellow
            (0, 255, 0),    # 3 - Green
            (255, 0, 0),    # 4 - Blue
            (128, 0, 128),  # 5 - Purple
            (0, 255, 255),  # 6 - Cyan
            (128, 0, 128)   # 7 - Magenta
        ]
        
        # Different colors for visible vs not visible
        self.visible_color = (0, 255, 0)    # Green
        self.not_visible_color = (0, 0, 255) # Red
        
        self.current_keypoint = 0
        self.keypoints = []  # List of (x, y, visibility) for each keypoint
        self.drawing = False
        
        # Initialize keypoints buffer for all images
        self.keypoints_buffer = {}  # Dictionary to store keypoints for each image
        
        # Find all image files in the data directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        for ext in image_extensions:
            self.image_files.extend(list(self.data_dir.glob(f'*{ext}')))
            self.image_files.extend(list(self.data_dir.glob(f'*{ext.upper()}')))
        
        # Sort files for consistent ordering
        self.image_files = sorted(self.image_files)
        
        if not self.image_files:
            print(f"No image files found in {self.data_dir}")
            sys.exit(1)
        
        print(f"Found {len(self.image_files)} image files")
        
    def load_annotations(self):
        """Load existing annotations from JSON file if available, or from buffer"""
        if self.current_image_idx < len(self.image_files):
            image_path = self.image_files[self.current_image_idx]
            json_path = image_path.with_suffix('.json')
            
            # Check if we have annotations in buffer for this image
            image_str = str(image_path)
            if image_str in self.keypoints_buffer:
                # Load from buffer
                self.keypoints = self.keypoints_buffer[image_str].copy()
                print(f"Loaded annotations from buffer for {image_path.name}")
            elif json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        # Look for the annotation for this specific image
                        if 'annotations' in data and len(data['annotations']) > 0:
                            # Get the first annotation
                            ann = data['annotations'][0]
                            # keypoints format is [x1,y1,v1,x2,y2,v2,...]
                            kp_values = ann.get('keypoints', [])
                            # Convert to our format [(x, y, v), ...]
                            self.keypoints = []
                            for i in range(0, len(kp_values), 3):
                                if i + 2 < len(kp_values):
                                    self.keypoints.append((kp_values[i], kp_values[i+1], kp_values[i+2]))
                                else:
                                    self.keypoints.append((-1, -1, 0))  # Default if incomplete
                            print(f"Loaded existing annotations from {json_path}")
                            # Also store in buffer
                            self.keypoints_buffer[image_str] = self.keypoints.copy()
                        else:
                            # Initialize with empty keypoints
                            self.keypoints = [(-1, -1, 0) for _ in range(8)]
                            # Initialize buffer for this image
                            self.keypoints_buffer[image_str] = self.keypoints.copy()
                    return True
                except Exception as e:
                    print(f"Error loading annotations: {e}")
            else:
                # Initialize with empty keypoints if no existing annotations
                self.keypoints = [(-1, -1, 0) for _ in range(8)]
                # Initialize buffer for this image
                self.keypoints_buffer[image_str] = self.keypoints.copy()
                return False
        else:
            # Initialize with empty keypoints if no existing annotations
            self.keypoints = [(-1, -1, 0) for _ in range(8)]
            return False
    
    def save_annotations(self):
        """Save annotations in COCO format from buffer"""
        if self.current_image_idx >= len(self.image_files):
            return
            
        image_path = self.image_files[self.current_image_idx]
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        
        # Get keypoints from buffer for this image
        image_str = str(image_path)
        if image_str in self.keypoints_buffer:
            keypoints_to_save = self.keypoints_buffer[image_str]
        else:
            keypoints_to_save = self.keypoints
        
        # Create COCO format annotation
        # Convert keypoints to flat list [x1,y1,v1,x2,y2,v2,...]
        keypoints_flat = []
        for x, y, v in keypoints_to_save:
            keypoints_flat.extend([x, y, v])
        
        # Create the COCO structure
        coco_data = {
            "images": [{
                "id": 0,
                "file_name": image_path.name,
                "width": width,
                "height": height
            }],
            "annotations": [{
                "id": 0,
                "image_id": 0,
                "category_id": 0,
                "keypoints": keypoints_flat,
                "num_keypoints": sum(1 for x, y, v in keypoints_to_save if v > 0),
                "bbox": self.calculate_bbox(keypoints_flat)
            }],
            "categories": [{
                "id": 0,
                "name": "volleyball_court",
                "keypoints": self.keypoints_names,
                "skeleton": [
                    [0, 4], [4, 1], [4, 6], [1, 2], 
                    [2, 5], [5, 6], [6, 7], [3, 0]
                ]
            }]
        }
        
        json_path = image_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Saved annotations to {json_path}")
    
    def calculate_bbox(self, keypoints_flat):
        """Calculate bounding box from keypoints [x1,y1,v1,x2,y2,v2,...]"""
        visible_points = []
        for i in range(0, len(keypoints_flat), 3):
            x, y, v = keypoints_flat[i], keypoints_flat[i+1], keypoints_flat[i+2]
            if v > 0:  # Only visible points
                visible_points.append([x, y])
        
        if not visible_points:
            return [0, 0, 1, 1]  # Default small bbox
        
        visible_points = np.array(visible_points)
        min_x = int(np.min(visible_points[:, 0]))
        min_y = int(np.min(visible_points[:, 1]))
        max_x = int(np.max(visible_points[:, 0]))
        max_y = int(np.max(visible_points[:, 1]))
        
        width = max_x - min_x
        height = max_y - min_y
        
        return [min_x, min_y, width, height]
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for handling clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click to set current keypoint
            self.keypoints[self.current_keypoint] = (x, y, 2)  # 2 = labeled and visible
            # Update buffer with the new keypoint
            image_path = str(self.image_files[self.current_image_idx])
            self.keypoints_buffer[image_path] = self.keypoints.copy()
            print(f"Set keypoint '{self.keypoints_names[self.current_keypoint]}' at ({x}, {y})")
            # Redraw the image to show the new keypoint immediately
            if self.current_image is not None:
                annotated_img = self.draw_annotations(self.current_image)
                cv2.imshow('Volleyball Court Annotation', annotated_img)
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Middle click to delete current keypoint
            self.keypoints[self.current_keypoint] = (-1, -1, 0)  # Reset to default
            # Update buffer with the deleted keypoint
            image_path = str(self.image_files[self.current_image_idx])
            self.keypoints_buffer[image_path] = self.keypoints.copy()
            print(f"Deleted keypoint '{self.keypoints_names[self.current_keypoint]}'")
            # Redraw the image to show the change immediately
            if self.current_image is not None:
                annotated_img = self.draw_annotations(self.current_image)
                cv2.imshow('Volleyball Court Annotation', annotated_img)
    
    def draw_annotations(self, image):
        """Draw keypoints and connections on image"""
        img_copy = image.copy()
        
        # Draw skeleton connections
        skeleton = [
            [0, 4], [4, 1], [4, 6], [1, 2], 
            [2, 5], [5, 3], [6, 7], [3, 0],
            [4,6],[7,5],[4,5]
        ]
        
        for conn in skeleton:
            pt1_idx, pt2_idx = conn
            pt1 = self.keypoints[pt1_idx]
            pt2 = self.keypoints[pt2_idx]
            
            # Only draw if both points are set
            if pt1[2] > 0 and pt2[2] > 0:  # Both visible
                cv2.line(img_copy, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), 
                        (200, 200, 200), 2)
        
        # Draw keypoints
        for i, (x, y, v) in enumerate(self.keypoints):
            if v > 0:  # Only draw if point is set
                color = self.visible_color if v == 2 else self.not_visible_color
                # Draw point
                cv2.circle(img_copy, (int(x), int(y)), 8, color, -1)
                cv2.circle(img_copy, (int(x), int(y)), 8, (255, 255, 255), 2)  # White border
                
                # Draw keypoint number
                cv2.putText(img_copy, str(i), (int(x)+10, int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img_copy, str(i), (int(x)+10, int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Highlight current keypoint
        if self.current_keypoint < len(self.keypoints):
            curr_x, curr_y, curr_v = self.keypoints[self.current_keypoint]
            if curr_v > 0:
                cv2.circle(img_copy, (int(curr_x), int(curr_y)), 12, (0, 255, 255), 3)  # Yellow highlight
        
        # Add info text
        h, w = img_copy.shape[:2]
        cv2.putText(img_copy, f"Image: {self.current_image_idx + 1}/{len(self.image_files)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_copy, f"Image: {self.current_image_idx + 1}/{len(self.image_files)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        cv2.putText(img_copy, f"Keypoint: {self.current_keypoint} - {self.keypoints_names[self.current_keypoint]}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_copy, f"Keypoint: {self.current_keypoint} - {self.keypoints_names[self.current_keypoint]}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Add instructions
        instructions = [
            "Controls:",
            "LMB - Set current keypoint",
            "MMB - Delete current keypoint",
            "N - Next keypoint",
            "P - Previous keypoint", 
            "SPACE - Skip keypoint",
            "S - Save annotations",
            '"]" - Next image',
            '"[" - Previous image',
            "ESC - Quit"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(img_copy, text, (w - 250, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img_copy, text, (w - 250, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img_copy
    
    def run(self):
        """Main annotation loop"""
        cv2.namedWindow('Volleyball Court Annotation', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Volleyball Court Annotation', self.mouse_callback)
        
        while True:
            # Load current image
            if self.current_image_idx < len(self.image_files):
                image_path = self.image_files[self.current_image_idx]
                self.current_image = cv2.imread(str(image_path))
                
                if self.current_image is None:
                    print(f"Could not load image: {image_path}")
                    continue
                
                # Load existing annotations or initialize
                self.load_annotations()
                
                # Draw annotations on image
                annotated_img = self.draw_annotations(self.current_image)
                
                # Show image
                cv2.imshow('Volleyball Court Annotation', annotated_img)
                
                # Wait for key press
                key = cv2.waitKey(0) & 0xFF
                
                if key == 27:  # ESC key
                    break
                elif key == ord('n') or key == ord('N'):  # Next keypoint
                    self.current_keypoint = (self.current_keypoint + 1) % 8
                elif key == ord('p') or key == ord('P'):  # Previous keypoint
                    self.current_keypoint = (self.current_keypoint - 1) % 8
                elif key == ord(']'):  # Next image
                    if self.current_image_idx < len(self.image_files) - 1:
                        self.current_image_idx += 1
                elif key == ord('['):  # Previous image
                    if self.current_image_idx > 0:
                        self.current_image_idx -= 1
                elif key == ord(' '):  # Space - skip keypoint
                    # Mark current keypoint as not visible (0)
                    if self.current_keypoint < len(self.keypoints):
                        x, y, v = self.keypoints[self.current_keypoint]
                        self.keypoints[self.current_keypoint] = (x, y, 0)
                        print(f"Skipped keypoint '{self.keypoints_names[self.current_keypoint]}'")
                        self.current_keypoint = (self.current_keypoint + 1) % 8
                elif key == ord('s') or key == ord('S'):  # Save
                    self.save_annotations()
                elif key == ord('m') or key == ord('M'):  # Toggle mode (keypoint/image navigation)
                    # This would toggle between navigating keypoints vs images
                    # For now, let's just add a trackbar to distinguish modes
                    pass
            else:
                break
        
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Volleyball Court Keypoints Annotation Tool')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Directory containing images to annotate (default: data)')
    
    args = parser.parse_args()
    
    annotator = VolleyballCourtAnnotator(args.data_dir)
    annotator.run()

if __name__ == "__main__":
    main()