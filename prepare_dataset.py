"""
Fruit Quality Detection - Dataset Preparation Script
Organizes raw fruit images (raw_dataset/<ClassName>/*.jpg) into a
train/validation-ready structure under dataset/ that
ImageDataGenerator.flow_from_directory() can consume directly.

Expected input layout:
    raw_dataset/
        Good Orange/
        Bad Orange/
        Good Apple/
        Bad Apple/
        Good Pomegranate/
        Bad Pomegranate/

Output layout:
    dataset/
        Good Orange/
        Bad Orange/
        ...
"""

import os
import shutil
import random

RAW_DIR = "raw_dataset"
OUT_DIR = "dataset"
SPLIT_RATIO = 0.8  # 80% train, handled further by validation_split during training

random.seed(42)


def prepare():
    if not os.path.isdir(RAW_DIR):
        print(f"Error: '{RAW_DIR}' folder not found.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    for class_name in os.listdir(RAW_DIR):
        class_path = os.path.join(RAW_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        out_class_path = os.path.join(OUT_DIR, class_name)
        os.makedirs(out_class_path, exist_ok=True)

        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)

        for img_name in images:
            src = os.path.join(class_path, img_name)
            dst = os.path.join(out_class_path, img_name)
            shutil.copy2(src, dst)

        print(f"Copied {len(images)} images for class '{class_name}'.")

    print("Dataset preparation complete. Output in:", OUT_DIR)


if __name__ == "__main__":
    prepare()
