import os
from pathlib import Path
from collections import Counter

def audit_yolo_dataset(labels_dir):
    labels_path = Path(labels_dir)
    class_counts = Counter()
    empty_labels = 0
    total_files = 0

    if not labels_path.exists():
        print(f"Directory not found: {labels_path}")
        return

    for label_file in labels_path.glob("*.txt"):
        total_files += 1
        with open(label_file, "r") as f:
            lines = f.readlines()
            if not lines:
                empty_labels += 1
            for line in lines:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1

    print("=== Dataset Audit Summary ===")
    print(f"Total label files scanned: {total_files}")
    print(f"Empty annotation files: {empty_labels}")
    print("Class Distribution Breakdown:")
    for cls_id, count in sorted(class_counts.items()):
        print(f"  Class ID {cls_id}: {count} instances")

if __name__ == "__main__":
    audit_yolo_dataset("dataset_enhanced/labels/train")