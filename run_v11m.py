from ultralytics import YOLO

# Лучший выбор: yolo11m.pt или yolo11x.pt (баланс точность/скорость)
model = YOLO("yolo11m-pose.pt")  # Или yolo11x.pt для max точности

data_yml = '/home/nssd/gled/vb/dataset-vb/vb-my-court/volleyball-court-keypoints-1/data.yaml'

data_yml = '/home/nssd/gled/vb/dataset-vb/vb-my-court/vb-backline-mycams/yolo-datasets/data.yaml'
data_yml = 'data/yolo-datasets/data.yaml'
#results = model.train(
#     data=data_yml,
#
#     epochs=150,                  # Custom kp требует больше эпох
#     imgsz=1280,                  # Сохраняем детали корта
#     batch=4,                    # Auto-batch (или 8-16 вручную)
#     device=0,                    # GPU
#     patience=50,
#     augment=True,                # Критично для вариаций кортов (освещение, угол)
#     hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
#     degrees=20,
#     translate=0.2, 
#     scale=0.9, shear=10.0, fliplr=0.5,
#     mosaic=1.0, mixup=0.5,      # Сильные аугментации для generalization
#     optimizer="AdamW",
#     lr0=0.001,
#     close_mosaic=20,
#     name="volleyball_court_yolo11m_pose_mosaic_bl",
#     project="runs/yolo11m_pose",
#     exist_ok=True
# )

model = YOLO("yolo11m-pose.pt")  # Или yolo11x.pt для max точности
results = model.train(
    data=data_yml,
    epochs=200,                  # Custom kp требует больше эпох
    imgsz=1280,                  # Сохраняем детали корта
    batch=6,                    # Auto-batch (или 8-16 вручную)
    device=0,                    # GPU
    patience=40,
    augment=True,                # Критично для вариаций кортов (освещение, угол)
    hsv_h=0.01,
    hsv_s=0.5,
    hsv_v=0.4,
    degrees=10,
    translate=0.05, 
    scale=0.2, shear=5.0,
    fliplr=0.5,
    mosaic=0.0,
    close_mosaic=0,
    mixup=0.0,      # Сильные аугментации для generalization
    optimizer="AdamW",
    lr0=0.0008,
    lrf=0.01,
    weight_decay=0.01,
    
    rect=True,          # сохраняет пропорции кадра
    cos_lr=True,        # плавный decay
    warmup_epochs=3,
    
    
    name="volleyball_court_yolo11m_pose_bl_work",
    project="runs/yolo11m_pose",
    exist_ok=True
)
