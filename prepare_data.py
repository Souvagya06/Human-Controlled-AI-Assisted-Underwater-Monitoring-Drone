import os
import json
import shutil
from pathlib import Path

# Paths based on your exact folder structure
TRAIN_JSON = "dataset/annotations/instances_train_trashcan.json"
VAL_JSON = "dataset/annotations/instances_val_trashcan.json"
IMAGES_DIR = "original_data/images"
OUTPUT_DIR = "dataset"

def process_trashcan():
    print("Starting Dataset Discovery, COCO to YOLO Conversion, and Structuring...")
    
    categories_list = []
    
    for split_type, json_path in [('train', TRAIN_JSON), ('val', VAL_JSON)]:
        if not os.path.exists(json_path):
            print(f"Skipping {split_type}, JSON not found at {json_path}")
            continue

        with open(json_path, 'r') as f:
            coco = json.load(f)

        # Capture categories dynamically from metadata for yaml generation
        if not categories_list and 'categories' in coco:
            categories_list = sorted(coco['categories'], key=lambda x: x['id'])

        images_dict = {img['id']: img for img in coco['images']}
        
        img_to_anns = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        cat_mapping = {cat['id']: i for i, cat in enumerate(sorted(coco['categories'], key=lambda x: x['id']))}

        os.makedirs(f"{OUTPUT_DIR}/images/{split_type}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split_type}", exist_ok=True)

        for img_id, img_info in images_dict.items():
            file_name = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']

            src_img_path = os.path.join(IMAGES_DIR, file_name)
            dst_img_path = os.path.join(OUTPUT_DIR, "images", split_type, file_name)
            
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)
            else:
                # Fallback: search recursively if flat path fails
                found_imgs = list(Path(IMAGES_DIR).rglob(file_name))
                if found_imgs:
                    shutil.copy(found_imgs[0], dst_img_path)

            label_filename = Path(file_name).stem + ".txt"
            label_path = os.path.join(OUTPUT_DIR, "labels", split_type, label_filename)

            yolo_lines = []
            if img_id in img_to_anns:
                for ann in img_to_anns[img_id]:
                    cat_id = ann['category_id']
                    class_idx = cat_mapping.get(cat_id, 0)
                    
                    bbox = ann['bbox']
                    x_min, y_min, w, h = bbox

                    # COCO to YOLO normalized calculation
                    x_center = (x_min + w / 2.0) / img_width
                    y_center = (y_min + h / 2.0) / img_height
                    norm_w = w / img_width
                    norm_h = h / img_height

                    # Strict Coordinate Bounds Checking
                    x_center = min(max(x_center, 0.0), 1.0)
                    y_center = min(max(y_center, 0.0), 1.0)
                    norm_w = min(max(norm_w, 0.0), 1.0)
                    norm_h = min(max(norm_h, 0.0), 1.0)

                    if norm_w > 0 and norm_h > 0:
                        yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            with open(label_path, 'w') as lf:
                lf.writelines(yolo_lines)

    # Automatically generate dataset.yaml using discovered categories
    if categories_list:
        class_names = [cat['name'] for cat in categories_list]
        yaml_content = f"""path: {Path(OUTPUT_DIR).absolute()}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
        yaml_path = "dataset.yaml"
        with open(yaml_path, 'w') as yf:
            yf.write(yaml_content)
        print(f"Automated dataset.yaml successfully created with {len(class_names)} classes!")

    print("Annotations successfully converted, directory structured, and images populated into YOLO format!")

if __name__ == '__main__':
    process_trashcan()