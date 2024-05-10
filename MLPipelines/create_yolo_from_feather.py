import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow.feather as feather

def make_segmentation(width, height, row):
    yolo_segmentation = ' '.join([f"{x / width} {y / height}" for x, y in zip(row.segmentation[::2], row.segmentation[1::2])]).split(' ')
    return yolo_segmentation

def make_bbox(width, height, row):
    scale = np.array([1.0/width, 1.0/height]).reshape(1, -1)
    poly = row.segmentation.reshape(-1, 2)
    poly = poly * scale
    x1 = np.min(poly[:, 0])
    y1 = np.min(poly[:, 1])
    x2 = np.max(poly[:, 0])
    y2 = np.max(poly[:, 1])
    # Format is center, width, height, normalized to image size
    aa_bbox = [(x1+x2)/2, (y2+y1)/2, x2-x1, y2-y1]
    return aa_bbox

def write_yolo_dataset(anno_df: pd.DataFrame, images_df: pd.DataFrame, cat_map: Dict, output_txt_dir: str, segmentation: bool = False) -> Dict:
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)

    # Collect images by filename
    image_map = {}
    image_names = set(anno_df['image_name'])
    for _, row in images_df.iterrows():
        img_name = row.image_name
        if not img_name in image_names:
            continue
        image_map[img_name] = {
            "anno": [],
            "width": row.width,
            "height": row.height
        }
    # Group Annotations by Filename
    for _, row in anno_df.iterrows():
        image_rec = image_map[row.image_name]
        # Drop the last point, since it is the same as the first point, expressed as a "segmentation"
        rec = image_rec['anno']

        if segmentation:
            box = make_segmentation(image_rec['width'], image_rec['height'], row)
        else:
            box = make_bbox(image_rec['width'], image_rec['height'], row)
        
        rec.append({
            'box': box,
            'category': cat_map[row.category]
        })

    # For each image write out the corresponding annotation
    for image_name in image_map:
        # Write out the image
        image_rec = image_map[image_name]
        txt_fn = os.path.join(
            output_txt_dir, image_name+".txt")
        with open(txt_fn, "w") as f:
            for anno in image_rec['anno']:
                box = " ".join(str(x) for x in anno['box'])
                anno_line = f"{anno['category']} {box}\n"
                f.write(anno_line)
            f.flush()
    print("Processed rows ", anno_df.shape)
    print("Images ", len(image_map))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train an object detection model.')
    parser.add_argument('--input_anno', type=str,
                        help='feather file with annotations')
    parser.add_argument('--input_image', type=str,
                        help='feather file with images')
    parser.add_argument('--output_txt_dir', type=str,
                        help='output file for the JSON')
    args = parser.parse_args()
    # Open the feather file
    anno_df = feather.read_feather(args.input_anno)
    images_df = feather.read_feather(args.input_image)
    write_yolo_dataset(anno_df, images_df, args.output_txt_dir)
