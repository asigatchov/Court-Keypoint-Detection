import os
import json
import argparse
from glob import glob


def find_image_file(base_path, image_name):
    for ext in ['.jpg', '.jpeg', '.png']:
        candidate = os.path.join(base_path, image_name + ext)
        if os.path.isfile(candidate):
            return os.path.basename(candidate)
    return None

def main(data_dir):
    merged = {
        'images': [],
        'annotations': [],
        'categories': [],
    }
    image_id_offset = 0
    annotation_id_offset = 0
    categories_map = {}
    categories_written = False

    json_files = [f for f in glob(os.path.join(data_dir, '*.json')) if not f.endswith('_annotations.coco.json')]
    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Categories (write once)
        if not categories_written and 'categories' in data:
            merged['categories'] = data['categories']
            for cat in data['categories']:
                categories_map[cat['id']] = cat
            categories_written = True
        # Images
        for img in data.get('images', []):
            img_name = os.path.splitext(img['file_name'])[0]
            found_img = find_image_file(data_dir, img_name)
            if found_img:
                new_img = img.copy()
                new_img['id'] = image_id_offset
                new_img['file_name'] = found_img
                merged['images'].append(new_img)
                # Annotations
                for ann in data.get('annotations', []):
                    if ann['image_id'] == img['id']:
                        new_ann = ann.copy()
                        new_ann['id'] = annotation_id_offset
                        new_ann['image_id'] = image_id_offset
                        merged['annotations'].append(new_ann)
                        annotation_id_offset += 1
                image_id_offset += 1

    with open(os.path.join(data_dir, '_annotations.coco.json'), 'w') as f:
        json.dump(merged, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge COCO JSONs with image check')
    parser.add_argument('--data_dir', required=True, help='Directory with JSON and image files')
    args = parser.parse_args()
    main(args.data_dir)
