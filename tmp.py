import pandas as pd
import cv2
import os

from tqdm import tqdm


def process(input_path, image_dir, output_path):
    rows = []

    with open(input_path, "r") as f:
        for line in tqdm(f):
            parts = line.strip().split(',')
            image_name = parts[0]
            n_landmarks = parts[1]
            coords = list(map(int, parts[2:]))
            
            # Load image and get size
            image_path = os.path.join(image_dir, image_name)
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                height, width = img.shape[:2]
            else:
                print(f"⚠️ Warning: {image_path} not found. Setting width/height to 0.")
                width, height = 0, 0

            row = [image_name, width, height, n_landmarks] + coords
            rows.append(row)

    # Generate column names
    header = ["image_name", "image_width", "image_height", "n_landmarks"]
    for i in range(len(rows[0][4:]) // 2):
        header += [f"landmark_{i+1}_x", f"landmark_{i+1}_y"]

    # Save to CSV
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved to {output_path}")


# Input/output paths
input_path = "data/txt/train_label.txt"
image_dir = "data/image/train"
output_path = "train_annotation_knee_long.csv"
process(input_path, image_dir, output_path)

input_path = "data/txt/test_label.txt"
image_dir = "data/image/test"
output_path = "test_annotation_knee_long.csv"
process(input_path, image_dir, output_path)