#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modular inference script for YOLOv11 pose model to detect different court types (tennis, badminton, volleyball) and visualize keypoints.
Calculates and displays FPS during video processing.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import argparse
import time


class CourtVisualizer:
    """Class to handle visualization for different court types"""
    
    @staticmethod
    def draw_tennis_keypoints(frame, keypoints, conf_threshold=0.5):
        """
        Draw tennis court keypoints and connections on the frame.
        
        Args:
            frame: Input image/frame
            keypoints: Keypoint coordinates from YOLO model
            conf_threshold: Minimum confidence threshold for keypoints
        """
        if keypoints is None:
            return frame
            
        # Extract keypoints and confidence
        kpts = keypoints.data[0]  # Shape: [num_keypoints, 3] where [x, y, confidence]
        
        # Define connections between keypoints to form the tennis court shape
        # These connections are based on tennis court geometry
        # Standard tennis court has 4 corners for outer rectangle and 4 for inner rectangle
        court_connections = [
            # Outer court boundaries (keypoints 0-3 are outer corners)
            (0, 1), (1, 3), (3, 2), (2, 0),  # Outer rectangle
            # Inner court boundaries (keypoints 4-7 are inner corners)
            (4, 6), (6, 7), (7, 5), (5, 4),  # Inner rectangle
            # Service line (keypoints 8-11 are service line points)
            (8, 9), (10, 11),  # Service lines
            # Net line (keypoints 12-13 are net top and bottom center)
            (12, 13),  # Net center line
        ]
        
        # Draw connections between keypoints
        for start_idx, end_idx in court_connections:
            if start_idx < len(kpts) and end_idx < len(kpts):
                start_conf = kpts[start_idx][2]
                end_conf = kpts[end_idx][2]
                
                if start_conf > conf_threshold and end_conf > conf_threshold:
                    start_point = (int(kpts[start_idx][0]), int(kpts[start_idx][1]))
                    end_point = (int(kpts[end_idx][0]), int(kpts[end_idx][1]))
                    
                    # Draw different colors for different court elements
                    if (start_idx <= 3 and end_idx <= 3):  # Outer court lines
                        color = (0, 255, 0)  # Green for outer court
                        thickness = 3
                    elif (4 <= start_idx <= 7 and 4 <= end_idx <= 7):  # Inner court lines
                        color = (0, 0, 255)  # Red for inner court
                        thickness = 3
                    elif (start_idx in [8, 9, 10, 11] or end_idx in [8, 9, 10, 11]):  # Service lines
                        color = (255, 0, 255)  # Magenta for service lines
                        thickness = 2
                    else:  # Net line
                        color = (255, 255, 0)  # Light blue for net
                        thickness = 2
                    
                    cv2.line(frame, start_point, end_point, color, thickness)
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(kpts):
            if conf > conf_threshold:
                # Draw different colors for different court elements
                if i <= 3:  # Outer court corners
                    color = (0, 255, 0)  # Green for outer court
                elif i <= 7:  # Inner court corners
                    color = (0, 0, 255)  # Red for inner court
                elif i <= 11:  # Service line points
                    color = (255, 0, 255)  # Magenta for service points
                else:  # Net or other points
                    color = (255, 255, 0)  # Light blue for net points
                
                # Draw keypoint
                cv2.circle(frame, (int(x), int(y)), 8, color, -1)
                # Label keypoint number
                cv2.putText(frame, str(i), (int(x)+10, int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    @staticmethod
    def draw_badminton_keypoints(frame, keypoints, conf_threshold=0.5):
        """
        Draw badminton court keypoints and connections on the frame.
        
        Args:
            frame: Input image/frame
            keypoints: Keypoint coordinates from YOLO model
            conf_threshold: Minimum confidence threshold for keypoints
        """
        if keypoints is None:
            return frame
            
        # Extract keypoints and confidence
        kpts = keypoints.data[0]  # Shape: [num_keypoints, 3] where [x, y, confidence]
        
        # For badminton, we have 30 keypoints (as per badminton_data.yml)
        # Define connections based on badminton court geometry using the provided skeleton
        court_connections = [
            (0, 1),(1,2), (2,3), (3,4),
            (5,6), (6,7), (7,8), (8,9),
            (4,9),(9,10),(10,19),(19,20),(20,29),   # Outer rectangle
            (29,28),  (28,27),(27,26),(26,25),
            (25,24),(24,15),(15,14),(14,5),(5,0),
            (14, 13), (13,12), (12,11),(11,10), 
            (15,16), (16,17), (17,18),(18,19)
        ]
        
        # Draw connections between keypoints
        for start_idx, end_idx in court_connections:
            if start_idx < len(kpts) and end_idx < len(kpts):
                start_conf = kpts[start_idx][2]
                end_conf = kpts[end_idx][2]
                
                if start_conf > conf_threshold and end_conf > conf_threshold:
                    start_point = (int(kpts[start_idx][0]), int(kpts[start_idx][1]))
                    end_point = (int(kpts[end_idx][0]), int(kpts[end_idx][1]))
                    
                    # Draw different colors for different court elements
                    if (start_idx <= 3 and end_idx <= 3):  # Outer court lines
                        color = (0, 255, 0)  # Green for outer court
                        thickness = 3
                    elif (4 <= start_idx <= 7 and 4 <= end_idx <= 7):  # Inner court lines
                        color = (0, 0, 255)  # Red for inner court
                        thickness = 3
                    elif (8 <= start_idx <= 11 or 8 <= end_idx <= 11):  # Service lines
                        color = (255, 0, 255)  # Magenta for service lines
                        thickness = 2
                    else:  # Net line
                        color = (255, 255, 0)  # Light blue for net
                        thickness = 2
                    
                    cv2.line(frame, start_point, end_point, color, thickness)
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(kpts):
            if conf > conf_threshold:
                # Draw different colors for different court elements
                if i <= 3:  # Outer court corners
                    color = (0, 255, 0)  # Green for outer court
                elif i <= 7:  # Inner court corners
                    color = (0, 0, 255)  # Red for inner court
                elif i <= 11:  # Service line points
                    color = (255, 0, 255)  # Magenta for service points
                else:  # Net or other points
                    color = (255, 255, 0)  # Light blue for net points
                
                # Draw keypoint
                cv2.circle(frame, (int(x), int(y)), 8, color, -1)
                # Label keypoint number
                cv2.putText(frame, str(i), (int(x)+10, int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    @staticmethod
    def draw_volleyball_keypoints(frame, keypoints, conf_threshold=0.5):
        """
        Draw volleyball court keypoints and connections on the frame (placeholder).
        
        Args:
            frame: Input image/frame
            keypoints: Keypoint coordinates from YOLO model
            conf_threshold: Minimum confidence threshold for keypoints
        """
        if keypoints is None:
            return frame
            
        # Extract keypoints and confidence
        kpts = keypoints.data[0]  # Shape: [num_keypoints, 3] where [x, y, confidence]
        
        # Placeholder for volleyball court - to be implemented later
        # For now, just draw basic court boundaries (assuming similar to tennis)
        court_connections = [
            [0, 4], [4, 1], [1, 2], 
            [2, 5], [5, 3], [3, 0],
            [4,5],
            [4,6],
            [6,7],
            [7,5]   
           
        ]
        
        # Draw connections between keypoints
        for start_idx, end_idx in court_connections:
            if start_idx < len(kpts) and end_idx < len(kpts):
                start_conf = kpts[start_idx][2]
                end_conf = kpts[end_idx][2]
                
                if start_conf > conf_threshold and end_conf > conf_threshold:
                    start_point = (int(kpts[start_idx][0]), int(kpts[start_idx][1]))
                    end_point = (int(kpts[end_idx][0]), int(kpts[end_idx][1]))
                    
                    # Draw different colors for different court elements
                    if (start_idx <= 3 and end_idx <= 3):  # Outer court lines
                        color = (0, 255, 0)  # Green for outer court
                        thickness = 3
                    else:  # Center line
                        color = (0, 0, 255)  # Red for center line
                        thickness = 2
                    
                    cv2.line(frame, start_point, end_point, color, thickness)
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(kpts):
            if conf > conf_threshold:
                # Draw different colors for different court elements
                if i <= 3:  # Outer court corners
                    color = (0, 255, 0)  # Green for outer court
                else:  # Center line or other points
                    color = (0, 0, 255)  # Red for other points
                
                # Draw keypoint
                cv2.circle(frame, (int(x), int(y)), 8, color, -1)
                # Label keypoint number
                cv2.putText(frame, str(i), (int(x)+10, int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame


def process_video(model_path, video_path, court_type='tennis', visualize=False, device='auto'):
    """
    Process video with YOLOv11 pose model and visualize keypoints based on court type.
    
    Args:
        model_path: Path to the trained model weights
        video_path: Path to input video
        court_type: Type of court ('tennis', 'badminton', 'volleyball')
        visualize: Whether to show the video in real-time
    """
    # Load the trained YOLOv11 pose model
    model = YOLO(model_path)
    
    # Set device based on user preference
    if device == 'cpu':
        model.to('cpu')
        print("Using CPU for inference")
    elif device == 'gpu' or device == 'cuda':
        if torch.cuda.is_available():
            model.to('cuda')
            print("Using GPU for inference")
        else:
            print("GPU requested but not available, using CPU")
            model.to('cpu')
    else:  # auto
        if torch.cuda.is_available():
            model.to('cuda')
            print("Auto-detected GPU, using GPU for inference")
        else:
            model.to('cpu')
            print("No GPU available, using CPU for inference")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0
    frame_count = 0
    paused = False  # Initialize pause state
    
    # Select the appropriate draw function based on court type
    if court_type == 'tennis':
        draw_func = CourtVisualizer.draw_tennis_keypoints
        window_title = f'Tennis Court Detection - YOLOv11 Pose'
    elif court_type == 'badminton':
        draw_func = CourtVisualizer.draw_badminton_keypoints
        window_title = f'Badminton Court Detection - YOLOv11 Pose'
    elif court_type == 'volleyball':
        draw_func = CourtVisualizer.draw_volleyball_keypoints
        window_title = f'Volleyball Court Detection - YOLOv11 Pose'
    else:
        print(f"Unsupported court type: {court_type}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Perform inference
        results = model(frame, verbose=False)  # Suppress verbose output
        
        # Process results
        annotated_frame = frame.copy()
        
        # Draw keypoints for each detected pose
        for result in results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                annotated_frame = draw_func(annotated_frame, result.keypoints)
        
        # Calculate FPS
        new_frame_time = time.time()
        if prev_frame_time != 0:
            fps_calc = 1 / (new_frame_time - prev_frame_time)
        else:
            fps_calc = 0
        prev_frame_time = new_frame_time
        
        # Add FPS text to frame
        cv2.putText(annotated_frame, f'FPS: {fps_calc:.2f}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add frame count
        cv2.putText(annotated_frame, f'Frame: {frame_count}/{total_frames}', 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add court type to frame
        cv2.putText(annotated_frame, f'Court Type: {court_type}', 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame if visualize is True
        if visualize:
            cv2.imshow(window_title, annotated_frame)
            
            # Press 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar to pause/resume
                paused = not paused
                if paused:
                    print("Playback paused. Press SPACE to resume.")
                else:
                    print("Playback resumed.")
            
            # If paused, wait indefinitely until space is pressed again
            while paused and visualize:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord(' '):  # Spacebar to resume
                    paused = not paused
                    if not paused:
                        print("Playback resumed.")
                    break
        
        # Print FPS periodically
        if frame_count % 30 == 0:  # Print every 30 frames
            print(f"Processed frame {frame_count}/{total_frames}, Current FPS: {fps_calc:.2f}")
    
    # Release video capture and close windows
    cap.release()
    if visualize:
        cv2.destroyAllWindows()
    
    print(f"Processing completed. Total frames processed: {frame_count}")


def main():
    parser = argparse.ArgumentParser(description='Inference script for YOLOv11 pose model to detect different court types')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model weights (e.g., ./runs/yolo11_pose/tennis_court_yolo11_pose/weights/best.pt)')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--court_type', type=str, default='tennis',
                        choices=['tennis', 'badminton', 'volleyball'],
                        help='Type of court to detect: tennis, badminton, or volleyball (default: tennis)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the output in real-time')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'gpu', 'cuda', 'auto'],
                        help='Device to run inference on: cpu, gpu(cuda), or auto (default: auto)')
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_path}")
    print(f"Processing video: {args.video_path}")
    print(f"Court type: {args.court_type}")
    print(f"Visualization: {args.visualize}")
    print(f"Device: {args.device}")
    
    process_video(args.model_path, args.video_path, args.court_type, args.visualize, args.device)


if __name__ == '__main__':
    main()