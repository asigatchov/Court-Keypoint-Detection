import json
import shutil
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Обрезает значение в диапазон [min_val, max_val]"""
    return max(min_val, min(max_val, value))

def convert_coco_to_yolo_pose(
    coco_json_path: str | Path,
    images_dir: str | Path,
    labels_output_dir: str | Path,
    images_output_dir: str | Path | None = None
):
    """
    Конвертирует COCO JSON с keypoints в YOLO pose формат (.txt).
    Опционально копирует изображения в новую структуру.
    """
    coco_json_path = Path(coco_json_path)
    images_dir = Path(images_dir)
    labels_output_dir = Path(labels_output_dir)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    if images_output_dir:
        Path(images_output_dir).mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Словарь изображений
    images = {img['id']: img for img in data['images']}
    anns_by_image = {}
    for ann in data['annotations']:
        anns_by_image.setdefault(ann['image_id'], []).append(ann)

    valid_images = 0
    ignored_annotations = 0

    for img_id, img_info in tqdm(images.items(), desc=f"Конвертация {coco_json_path.parent.name}"):
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']

        print(img_info)

        src_img_path = images_dir / file_name
        if images_output_dir:
            dst_img_path = Path(images_output_dir) / file_name
            if not dst_img_path.exists():
                if src_img_path.exists():
                    shutil.copy2(src_img_path, dst_img_path)
                else:
                    print(f"Предупреждение: изображение не найдено: {src_img_path}")
                    continue

        label_path = labels_output_dir / (Path(file_name).stem + '.txt')
        lines = []

        for ann in anns_by_image.get(img_id, []):
            class_id = ann['category_id'] #- 1

            # Bbox: [x, y, w, h] → нормализованный центр + размеры
            x, y, w_box, h_box = ann['bbox']

            # Проверяем, чтобы bbox не выходил за границы изображения
            # Если bbox выходит за границы, корректируем его
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w_box)
            y2 = min(height, y + h_box)
            
            # Пересчитываем bbox, если он был скорректирован
            if x1 != x or y1 != y or x2 != x + w_box or y2 != y + h_box:
                w_box = x2 - x1
                h_box = y2 - y1
                x = x1
                y = y1

            # Нормализация
            x_center = (x + w_box / 2) / width
            y_center = (y + h_box / 2) / height
            norm_w = w_box / width
            norm_h = h_box / height
            
            # Clamp все значения в [0, 1] и проверка валидности
            x_center = clamp(x_center)
            y_center = clamp(y_center)
            norm_w = clamp(norm_w)
            norm_h = clamp(norm_h)
            
            # Защита от нулевого или невалидного bbox
            if norm_w <= 0 or norm_h <= 0 or x_center > 1.0 or y_center > 1.0:
                ignored_annotations += 1
                continue

            # Keypoints
            print(file_name)
            kpts_raw = ann['keypoints']
            normalized_kpts = []
            valid_keypoints = True
            for i in range(0, len(kpts_raw), 3):
                px, py, visibility = kpts_raw[i], kpts_raw[i+1], kpts_raw[i+2]
                # Handle visibility flag - 0=invisible, 1=visible but outside image, 2=visible inside image
                if visibility == 0:
                    # Invisible keypoints should have coordinates set to 0
                    normalized_kpts.extend([0.0, 0.0, 0.0])
                else:
                    nx = clamp(px / width)
                    ny = clamp(py / height)
                    # Check if normalized coordinates are valid
                    if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                        valid_keypoints = False
                        break
                    visibility = 1.0
                    normalized_kpts.extend([nx, ny, visibility])
            
            # Skip this annotation if any keypoints are invalid
            if not valid_keypoints:
                ignored_annotations += 1
                continue

            # Формируем строку с дополнительной проверкой
            parts = [str(class_id), f"{x_center:.6f}", f"{y_center:.6f}", f"{norm_w:.6f}", f"{norm_h:.6f}"]
            
            # Добавляем keypoints
            for i in range(0, len(normalized_kpts), 3):
                kx, ky, kvis = normalized_kpts[i], normalized_kpts[i+1], normalized_kpts[i+2]
                parts.extend([f"{kx:.6f}", f"{ky:.6f}", f"{kvis:.6f}"])
            
            # Делаем финальную проверку всех значений в строке
            line_str = " ".join(parts)
            values = [float(x) for x in line_str.split()]
            
            # Проверяем, что все bbox значения (x_center, y_center, width, height) в пределах [0,1]
            bbox_vals = values[1:5]  # class_id, x_center, y_center, width, height
            if any(v < 0 or v > 1 for v in bbox_vals):
                ignored_annotations += 1
                continue
            
            # Проверяем, что координаты keypoints (начиная с индекса 5, каждые 3 значения x, y, visibility)
            skip_annotation = False
            for i in range(5, len(values), 3):
                if i + 2 < len(values):  # Ensure we have x, y, visibility
                    kx, ky = values[i], values[i+1]
                    if kx < 0 or kx > 1 or ky < 0 or ky > 1:
                        ignored_annotations += 1
                        skip_annotation = True
                        break
            
            if skip_annotation:
                continue
            
            lines.append(line_str)

        # Записываем файл меток
        if lines:
            label_path.write_text("\n".join(lines) + "\n")
            valid_images += 1
        else:
            label_path.touch()  # пустой файл — YOLO поймёт, что аннотаций нет

    print(f"{coco_json_path.parent.name}: "
          f"валидных изображений с метками: {valid_images}, "
          f"проигнорировано аннотаций: {ignored_annotations}")

# ==============================================================================
# Запуск конвертации
# ==============================================================================

# Парсинг аргументов командной строк
def parse_args():
    parser = argparse.ArgumentParser(description="Конвертация COCO keypoints в YOLO pose формат")
    parser.add_argument("--data_dir", type=str, default="data", help="Директория с данными (по умолчанию: data)")
    return parser.parse_args()


def extract_dataset_info(coco_json_path):
    """Извлекает информацию о датасете из COCO JSON файла"""
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Извлекаем информацию о категориях
    categories = {cat['id']: cat for cat in data['categories']}
    
    # Извлекаем информацию о keypoints
    if data['categories']:
        # Берем информацию о keypoints из первой категории
        first_category = data['categories'][0]
        if 'keypoints' in first_category and 'skeleton' in first_category:
            keypoints_names = first_category.get('keypoints', [])
            skeleton = first_category.get('skeleton', [])
            num_keypoints = len(keypoints_names) if keypoints_names else 0
        else:
            # Если keypoints не определены в категории, попробуем определить из аннотаций
            sample_ann = data['annotations'][0] if data['annotations'] else None
            if sample_ann and 'keypoints' in sample_ann:
                # В COCO keypoints представлены в формате [x1,y1,v1,x2,y2,v2,...]
                num_keypoints = len(sample_ann['keypoints']) // 3
                keypoints_names = [f'keypoint_{i}' for i in range(num_keypoints)]
                skeleton = []
            else:
                num_keypoints = 0
                keypoints_names = []
                skeleton = []
    else:
        num_keypoints = 0
        keypoints_names = []
        skeleton = []
    
    return {
        'num_classes': len(categories),
        'class_names': {cat_id: cat['name'] for cat_id, cat in categories.items()},
        'num_keypoints': num_keypoints,
        'keypoints_names': keypoints_names,
        'skeleton': skeleton
    }


def create_data_yaml(dataset_info, output_path):
    """Создает YAML файл с информацией о датасете"""
    import yaml
    
    data_yaml = {
        'train': 'images/train',
        'val': 'images/valid',
        'nc': dataset_info['num_classes'],
        'names': [dataset_info['class_names'][i] for i in sorted(dataset_info['class_names'].keys())],
        'kpt_shape': [dataset_info['num_keypoints'], 3]  # 3 = (x, y, visibility)
    }
    
    # Добавляем информацию о keypoints, если они есть
    if dataset_info['keypoints_names']:
        data_yaml['keypoints'] = dataset_info['keypoints_names']
    
    if dataset_info['skeleton']:
        data_yaml['skeleton'] = dataset_info['skeleton']
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Файл конфигурации создан: {output_path}")


args = parse_args()
root = Path(args.data_dir)

# Удаляем старые labels и кэши (рекомендуется перед новой конвертацией)
for split in ["train", "valid"]:
    label_dir = root / "labels" / split
    if label_dir.exists():
        shutil.rmtree(label_dir)
    cache_file = root / "labels" / f"{split}.cache"
    if cache_file.exists():
        cache_file.unlink()

# Конвертация train
convert_coco_to_yolo_pose(
    coco_json_path=root / "train" / "_annotations.coco.json",
    images_dir=root / "train",
    labels_output_dir=root / "labels" / "train",
    images_output_dir=root / "images" / "train"   # Копируем изображения сюда
)

# Конвертация valid
convert_coco_to_yolo_pose(
    coco_json_path=root / "valid" / "_annotations.coco.json",
    images_dir=root / "valid",
    labels_output_dir=root / "labels" / "valid",
    images_output_dir=root / "images" / "valid"
)

# Извлекаем информацию о датасете из тренировочного COCO JSON файла
train_coco_path = root / "train" / "_annotations.coco.json"
if train_coco_path.exists():
    dataset_info = extract_dataset_info(train_coco_path)
    # Создаем data.yaml файл с информацией о датасете
    data_yaml_path = root / "data.yaml"
    create_data_yaml(dataset_info, data_yaml_path)
else:
    print(f"Предупреждение: файл {train_coco_path} не найден")

print("\nКонвертация завершена успешно!")
print("Структура готова для обучения YOLOv11 pose.")
print("Теперь можно запускать run_v11m.py")
