import argparse
import sys
import os
import json
import io
from typing import Dict
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import numpy as np
from PIL import Image


def make_anno_odtk(image_id, anno_id, cat_map, train, row):
    anno = {
        "iscrowd": int(False),
        "image_id": image_id,
        "bbox": list(row.rcoco),
        "category_id": int(cat_map[row.category]),
        "area": row.rcoco[2]*row.rcoco[3],
        "id": anno_id
    }
    # store segmentation only for the validation set
    if not train:
        anno['segmentation'] = [list(row.rbox)]
    return anno

def make_anno_d2(image_id, anno_id, cat_map, row):
    poly = row.segmentation.reshape(-1, 2)
    x1 = np.min(poly[:, 0])
    y1 = np.min(poly[:, 1])
    x2 = np.max(poly[:, 0])
    y2 = np.max(poly[:, 1])
    aa_bbox = [x1, y1, x2-x1, y2-y1]
    anno = {
        "iscrowd": int(False),
        "image_id": image_id,
        "bbox": aa_bbox,
        "category_id": int(cat_map[row.category]),
        "area": row.rcoco[2]*row.rcoco[3],
        "id": anno_id,
        "segmentation": [list(row.segmentation)]
    }
    return anno

def write_coco_dataset(anno_df: pd.DataFrame, images_df: pd.DataFrame, train: bool, output_json: str, odtk: bool = True) -> Dict:

    # COCO INFO
    info = {
        "description": "Dataset",
        "url": "http://permaling.com",
        "version": "1.0",
        "year": 2022,
        "contributor": "Permaling",
        "date_created": "2022/04/29"
    }

    # Catagories
    categories = []
    cat_map = {}
    cats = np.sort(anno_df['category'].unique())
    for i, cat in enumerate(cats):
        # The first index must be 1 because background is 0
        i += 1
        categories.append({
            "supercategory": cat,
            "id": i,
            "name": cat
        })
        cat_map[cat] = i

    # Images
    images = []
    image_map = {}
    image_id = 0
    for _, row in images_df.iterrows():
        images.append({
            "license": 1,
            "file_name": row.image_name+".jpeg",
            "height": row.height,
            "width": row.width,
            "id": image_id,
        })
        image_map[row.image_name] = image_id
        image_id += 1

    # Not really CC
    licenses = [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        }
    ]

    # Create an "eval" dataset so there is no augmentation
    annotations = []
    anno_id = 0
    for _, row in anno_df.iterrows():
        image_id = int(image_map[row.image_name])
        if odtk:
            anno = make_anno_odtk(image_id, anno_id, cat_map, train, row)
        else:
            anno = make_anno_d2(image_id, anno_id, cat_map, row)
        annotations.append(anno)
        anno_id += 1
    # Create the complete COCO JSON document
    coco_json = {
        "info": info,
        "licenses": licenses,
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(output_json, "w") as f:
        f.write(json.dumps(coco_json))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train an object detection model.')
    parser.add_argument('--input_anno', type=str,
                        help='feather file with annotations')
    parser.add_argument('--input_image', type=str,
                        help='feather file with images')
    parser.add_argument('--train', action='store_true',
                        help='this is a training set; when absent validation')
    parser.add_argument('--output_json', type=str,
                        help='output file for the JSON')
    args = parser.parse_args()
    # Open the feather file
    anno_df = feather.read_feather(args.input_anno)
    images_df = feather.read_feather(args.input_image)
    write_coco_dataset(anno_df, images_df, args.train, args.output_json)
