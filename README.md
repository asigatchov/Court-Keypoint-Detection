
# Volleyball Court Keypoint Detection with **YOLOv11** Pose

This repository provides a complete pipeline for training and running inference with a **YOLOv11 pose estimation model** to detect key points on a volleyball court. It focuses on back-view cameras in indoor gyms.

The model detects 8 key court keypoints and connects them to reconstruct court lines. This enables applications such as:

- Homography estimation and court warping
- Mapping player positions to court coordinates
- Ball trajectory analysis
- Broadcast graphics augmentation

### Example Detection

![Volleyball Court Annotator](https://raw.githubusercontent.com/asigatchov/Court-Keypoint-Detection/refs/heads/main/imgs/Volleyball_Court.jpg)

[Article Volleyball Court detection](https://recdev.ru/blog/5/)

## Features

- **Custom annotation tool** (vb-court_annotator.py) for manual keypoint labeling
- Dataset preparation scripts (split, merge COCO JSONs, convert to YOLO format)
- Training scripts for **YOLOv11m-pose** and **YOLOv11s-pose** with tuned hyperparameters
- Multi-court inference script (supports volleyball, tennis, badminton – easily extensible)
- Real-time visualization with FPS counter and pause/resume

## Keypoints Definition

The model uses 8 keypoints:

|ID|Name|Description|
|---|---|---|
|0|1_back_left|Back left corner (far side)|
|1|2_back_left|Adjacent back left point|
|2|3_back_right|Back right corner (far side)|
|3|4_back_right|Adjacent back right point|
|4|5_center_left|Attack line left|
|5|6_center_right|Attack line right|
|6|7_net_left|Net pole left|
|7|8_net_right|Net pole right|

Skeleton connections form the full court outline and net.

## Project Structure

text

```
.
├── vb-court_annotator.py          # Manual annotation tool
├── run_v11m.py                    # Training script (YOLOv11m-pose)
├── multi_court_inference.py       # Inference on video with visualization
├── coco2yolo_keypoints.py         # Convert COCO → YOLO pose format
├── split_dataset.py               # Split dataset into train/val
├── merge_jsons.py                 # Merge multiple COCO JSONs
├── pyproject.toml                 # Dependencies
├── data/                          # Place your images + annotations here
└── runs/                          # Training results (Ultralytics format)
```

## Setup

1. **Clone the repository**
    
    Bash
    
    ```
    git clone https://github.com/asigatchov/Court-Keypoint-Detection.git
    cd Court-Keypoint-Detection
    ```
    
2. **Install dependencies**
    
    Bash

    **with venv** 
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install ultralytics opencv-python tqdm

    ```

   **with uv tools**
    ```
    uv sync
    ```

## Workflow

### 1. Annotate Images

Use the interactive tool:

Bash

```
python vb-court_annotator.py --data_dir path/to/your/images
```

**Controls**:

- **LMB** — Set current keypoint
- **MMB** — Delete current keypoint
- **N / P** — Next / Previous keypoint
- **SPACE** — Skip keypoint (mark as invisible)
- **S** — Save annotation
- **[ / ]** — Previous / Next image
- **ESC** — Quit

Annotations save as COCO-style .json files next to each image.

### 2. Prepare Dataset for YOLO

Bash

```
# Split into train/valid
python split_dataset.py --input_dir path/to/annotated --output_dir yolo-datasets

# Merge COCO JSONs per split (if needed)
python merge_jsons.py --data_dir yolo-datasets/train/
python merge_jsons.py --data_dir yolo-datasets/valid/

# Convert to YOLO format + generate data.yaml
python coco2yolo_keypoints.py --data_dir yolo-datasets/
```

This creates the YOLO structure with images/, labels/, and data.yaml.

### 3. Train the Model

Edit run_v11m.py to point to your data.yaml, then:

Bash

```
python run_v11m.py
```

Start with yolo11s-pose.pt. Switch to yolo11m-pose.pt for higher accuracy.

Weights and logs appear in runs/yolo11m_pose/....

### 4. Inference on Video

Bash

```
python multi_court_inference.py \
  --model_path runs/yolo11m_pose/.../weights/best.pt \
  --video_path path/to/video.mp4 \
  --court_type volleyball \
  --visualize
```

Press **SPACE** to pause/resume, **q** to quit.

## Results

With 500–1000 annotated images from varied angles and lighting, the model achieves robust detection on challenging back views. Heavy augmentations ensure good generalization across gyms.

```bash
uv run split_dataset.py --input_dir /home/nssd/gled/vb/dataset-vb/vb-my-court/vb-backline-mycams/datasets --output_dir /home/nssd/gled/vb/dataset-vb/vb-my-court/vb-backline-mycams/yolo-datasets
uv run merge_jsons.py --data_dir /home/nssd/gled/vb/dataset-vb/vb-my-court/vb-backline-mycams/yolo-datasets/valid/
uv run merge_jsons.py --data_dir /home/nssd/gled/vb/dataset-vb/vb-my-court/vb-backline-mycams/yolo-datasets/train/
uv run coco2yolo_keypoints.py --data_dir /home/nssd/gled/vb/dataset-vb/vb-my-court/vb-backline-mycams/yolo-datasets/
```


