# Must be on Python Path
import io
import logging
import os
import random
import re
from typing import List, Any, Tuple, Dict

import fsspec
import gcsfs
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pac
import pyarrow.dataset as ds
import pyarrow.feather as feather
import pyarrow.fs as fs
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Wrapper class for fetching datasets from the datalake

rbb_anno_proj = {
    "project_id": ds.field("project_id").cast("int64"),
    "task_id": ds.field("task_id").cast("int64"),
    "job_id": ds.field("job_id").cast("int64"),
    "image_name": ds.field("image_name"),
    "category": ds.field("category"),
    "segmentation": ds.field("segmentation"),
    "rcoco": ds.field("rcoco"),
    "coco": ds.field("coco"),
    "gt_iid": ds.field("gt_iid"),
    "ts": ds.field("ts").cast("timestamp[us]"),
}

four_d_anno_proj = {
    "project_id": ds.field("project_id").cast("int64"),
    "task_id": ds.field("task_id").cast("int64"),
    "job_id": ds.field("job_id").cast("int64"),
    "image_name": ds.field("image_name"),
    "category": ds.field("category"),
    "segmentation": ds.field("segmentation"),
    "coco": ds.field("coco"),
    "report_id": ds.field("report_id"),
    "ts": ds.field("ts").cast("timestamp[us]"),
}

image_proj = {
    "project_id": ds.field("project_id").cast("int64"),
    "task_id": ds.field("task_id").cast("int64"),
    "job_id": ds.field("job_id").cast("int64"),
    "image_name": ds.field("image_name"),
    "image_bytes": ds.field("image_bytes"),
    "tags": ds.field("tags"),
    "ts": ds.field("ts").cast("timestamp[us]"),
}

aa_anno_proj = {
    "project_id": ds.field("project_id").cast("int64"),
    "task_id": ds.field("task_id").cast("int64"),
    "job_id": ds.field("job_id").cast("int64"),
    "image_name": ds.field("image_name"),
    "category": ds.field("category"),
    "bbox": ds.field("bbox"),
    "ts": ds.field("ts").cast("timestamp[us]"),
}

class DatasetObjDetect(object):
    # Schema for Labels

    anno_schema = pa.schema([
            pa.field('project_id', pa.int64()),
            pa.field('task_id', pa.int64()),
            pa.field('job_id', pa.int64()),
            pa.field('track_id', pa.int64()),
            pa.field('image_name', pa.string()),
            pa.field('category', pa.string()),
            pa.field('segmentation', pa.list_(pa.float32())),
            pa.field('rcoco', pa.list_(pa.float32())),
            pa.field('coco', pa.list_(pa.float32())),
            pa.field('gt_iid', pa.int64()),
            pa.field('gt_attr', pa.string()),
            pa.field('ts', pa.timestamp('us', tz='UTC')),
        ],
        metadata={
            "project_id": "ID of the project in CVAT",
            "task_id": "ID of the task in CVAT",
            "job_id": "ID of the job in CVAT",
            "track_id": "The track id, if labeled as tracks",
            "image_name": "The unique image name",
            "category": "Type of annotation e.g instrument, barcode, text, excluderegion, badimage",
            "segmentation": "The contour around the surgical instrument",
            "rcoco": "The position and dimensions of the bounding box including the rotation angle",
            "coco": "The position and dimensions of the bounding box",
            "gt_iid": "Item ID of the annotated instrument",
            "gt_attr": "String containing attributes and values in JSON format",
            "ts": "Time stamp of the annotation",
        }
    )

    four_d_anno_schema = pa.schema([
            pa.field('project_id', pa.int64()),
            pa.field('task_id', pa.int64()),
            pa.field('job_id', pa.int64()),
            pa.field('track_id', pa.int64()),
            pa.field('image_name', pa.string()),
            pa.field('category', pa.string()),
            pa.field('segmentation', pa.list_(pa.float32())),
            pa.field('coco', pa.list_(pa.float32())),
            pa.field('report_id', pa.string()),
            pa.field('ts', pa.timestamp('us', tz='UTC')),
        ],
        metadata={
            "project_id": "ID of the project in CVAT",
            "task_id": "ID of the task in CVAT",
            "job_id": "ID of the job in CVAT",
            "track_id": "The track id, if labeled as tracks",
            "image_name": "The unique image name",
            "category": "Type of annotation e.g instrument, barcode, text, excluderegion, badimage",
            "segmentation": "The contour around the surgical instrument",
            "coco": "The position and dimensions of the bounding box",
            "report_id": "The UUID from the report for the image",
            "ts": "Time stamp of the annotation",
        }
    )

    image_schema = pa.schema([
            pa.field('project_id', pa.int64()),
            pa.field('task_id', pa.int64()),
            pa.field('job_id', pa.int64()),
            pa.field('image_name', pa.string()),
            pa.field('image_bytes', pa.binary()), 
            pa.field('tags', pa.list_(pa.string())),
            pa.field('ts', pa.timestamp('us', tz='UTC')),
        ],
        metadata={
            "project_id": "ID of the project in CVAT",
            "task_id": "ID of the task in CVAT",
            "job_id": "ID of the job in CVAT",
            "image_name": "The unique image name",
            "image_bytes": "The image as a JPEG",
            "tags": "List of tags associated with the image",
            "ts": "Time stamp of the annotation",
        }
    )

    # Image names are our unit for splitting train and test,
    # so we need to partition on it
    anno_part_cols = ['project_id', 'job_id', 'image_name']

    # Schema for Images
    # Note: there are other fields on some projects, such as
    # task_id and tags

    image_part_cols = ['project_id', 'job_id', 'image_name']

    def __init__(self, project: str, token: str, bucket: str,
        dataset: str, anno_dir: str = "annotation", image_dir: str = "image", logger=None):
        self.dlfs = gcsfs.GCSFileSystem(project=project, token=token)
        self.pa_fs = fs.PyFileSystem(fs.FSSpecHandler(self.dlfs))
        self.anno_path = os.path.join(bucket, dataset, anno_dir)
        self.image_path = os.path.join(bucket, dataset, image_dir)
        if dataset == "od_rbb":
            self.anno_proj = rbb_anno_proj
        elif dataset == "4d_od":
            self.anno_proj = four_d_anno_proj
            self.anno_schema = self.four_d_anno_schema
        else:
            self.anno_proj = aa_anno_proj
        self.images = None
        self.annotations = None

    def add_annotations(self, anno: pd.DataFrame):
        pq.write_to_dataset(pa.Table.from_pandas(anno, schema=self.anno_schema), self.anno_path,
                            partition_cols=self.anno_part_cols, filesystem=self.dlfs, max_partitions=4096)

    def add_images(self, images: pd.DataFrame):
        pq.write_to_dataset(pa.Table.from_pandas(images, schema=self.image_schema), self.image_path,
                            partition_cols=self.image_part_cols, filesystem=self.dlfs, max_partitions=4096)

    def create_datasources(self):
        if self.images is None:
            self.images = ds.dataset(self.image_path, filesystem=self.pa_fs, schema=self.image_schema, partitioning="hive")
        if self.annotations is None:
            self.annotations = ds.dataset(self.anno_path, schema=self.anno_schema, filesystem=self.pa_fs, partitioning="hive")

    def unique_image_names(self, project_ids: List[int], skip_tags: List[str] = [], image_feather: str = None) -> List[str]:
        if image_feather is not None:
            image_df = []
            for project_id in project_ids:
                image_df.append(feather.read_feather(image_feather.format(project_id)))
            image_df = pd.concat(image_df)
        else:
            self.create_datasources()
            image_df = self.images.to_table(columns=["image_name", "tags"], filter=(ds.field("project_id").isin(project_ids))).to_pandas()
        image_df = image_df[image_df['tags'].apply(lambda x: self.filter_skip_tags(skip_tags, x))]
        return image_df['image_name'].unique()

    @staticmethod
    def filter_skip_tags(skip_tags, tags):
        t = list(tags)
        for skip_tag in skip_tags:
            if skip_tag in t:
                print("Skipping")
                return False
        return True
    
    def fetch_image_partitions(self) -> pd.DataFrame:
        self.create_datasources()
        # the image name is actually an image path, with '/' delimiters
        path_pat = re.compile("project_id\=(\d+)\/job_id\=([0-9]+)\/image_name\=(.+)/")
        project_ids = []
        job_ids = []
        image_names = []
        for row in self.images.files:
            m = path_pat.search(row)
            if m is None:
                logger.error(f"Could not parse path {row}")
                break
            project_ids.append(int(m.group(1)))
            job_ids.append(int(m.group(2)))
            image_names.append(m.group(3))
        df = pd.DataFrame({
            'project_id': project_ids,
            'job_id': job_ids,
            'image_name': image_names
        })
        return df.drop_duplicates()

    def image_sampler(self, project_ids: List[int], skip_tags: List[str] = [], image_feather: str = None, anno_feather: str = None, p: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = []
        validate_df = []
        image_names = self.unique_image_names(project_ids, skip_tags, image_feather)
        print('Found', len(image_names), 'unique images')
        np.random.shuffle(image_names)
        val_images = image_names[:int(len(image_names)*p)]
        anno_df = []
        if anno_feather is not None:
            anno_df = []
            for project_id in project_ids:
                anno_df.append(feather.read_feather(anno_feather.format(project_id)))
            anno_dfs = pd.concat(anno_df)
        for image_name in image_names:
            if anno_feather is not None:
                anno_df = anno_dfs[anno_dfs['image_name'] == image_name]
            else:
                anno_df = self.annotations.to_table(columns=self.anno_proj, filter=ds.field("image_name") == image_name).to_pandas()
            if image_name in val_images:
                print("Adding", image_name, "to val")
                validate_df.append(anno_df)
            else:
                print("Adding", image_name, "to train")
                train_df.append(anno_df)
        if train_df:
            train_df = pd.concat(train_df)
        if validate_df:
            validate_df = pd.concat(validate_df)
        return train_df, validate_df

    def get_annotations_by_project(self, project_id: int, skip_tags: List = []) -> pd.DataFrame:
        self.create_datasources()
        image_names = self.unique_image_names([project_id], skip_tags=skip_tags)
        anno_df = []
        for image_name in image_names:
            anno_df.append(self.annotations.to_table(columns=self.anno_proj, filter=ds.field("image_name") == image_name).to_pandas())
        if anno_df:
            anno_df = pd.concat(anno_df)
        return anno_df

    @staticmethod
    def redact_segmentation(img, cats, segs):

        def poly_points(seg):
            uvs = []
            for i in range(1, len(seg), 2):
                uvs.append((seg[i-1], seg[i]))
            uvs.append((seg[0], seg[1]))
            return uvs

        draw = ImageDraw.Draw(img)
        for cat, seg in zip(cats, segs):
            if cat == "excluderegion":
                draw.polygon(poly_points(seg), fill="black")
        return img

    @staticmethod
    def preprocess_image_with_labels(img, anno_df, row):
        labels = anno_df[anno_df['image_name'] == row.image_name]
        labs = DatasetObjDetect.redact_segmentation(img, labels['category'].values, labels['segmentation'].values)
        return img

    def write_images(self, anno_df: pd.DataFrame, output_dir: str, image_feather: str = None, image_dir: str = None) -> None:
        image_names = []
        widths = []
        heights = []
        tags = []
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        count = 0
        if image_feather is not None:
            image_df = []
            for project_id in anno_df['project_id'].unique():
                image_df.append(feather.read_feather(image_feather.format(project_id)))
            image_dfs = pd.concat(image_df)
        else:
            self.create_datasources()
        # Remove things labeled 'badimage'
        for image_name in anno_df['image_name'].unique():
            image_name = os.path.splitext(image_name)[0]
            if image_feather is not None:
                image_df = image_dfs[image_dfs['image_name'] == image_name]
            else:
                image_df = self.images.to_table(columns=image_proj, filter=ds.field("image_name") == image_name).to_pandas()
            tag_list = [tag for taglist in list(image_df['tags']) for tag in taglist]
            if 'badimage' in tag_list:
                print('Skipping bad image', image_name)
                continue
            for _, row in image_df.iterrows():
                if image_dir is not None and os.path.exists(os.path.join(image_dir, f"{image_name}.jpeg")):
                    img = Image.open(os.path.join(image_dir, f"{image_name}.jpeg"))
                else:
                    img = Image.open(io.BytesIO(row.image_bytes))
                # This will redact excluded regions
                img = self.preprocess_image_with_labels(img, anno_df, row)
                img.save(os.path.join(output_dir, f"{image_name}.jpeg"))
                image_names.append(image_name)
                widths.append(img.width)
                heights.append(img.height)
                tags.append(tag_list)
            if count % 10:
                print(".", end="")
            count += 1
        print()
        return pd.DataFrame({
            'image_name': image_names,
            'width': widths,
            'height': heights,
            'tags': tags
        })

class DatasetN1Crops(object):

    # TODO: remove `jpeg_image` once we're confident
    image_proj = {
        "item_id": ds.field("item_id").cast("int64"),
        "image_name": ds.field("image_name"),
        "crop_data": ds.field("crop_data"),
        "scene_name": ds.field("scene_name"),
        "crop_id": ds.field("crop_id"),
        "capture_uuid": ds.field("capture_uuid"),
        "ts": ds.field("ts").cast("timestamp[us]"),
    }

    # TODO: partition only by item_id - it will be faster
    image_part_cols = ['item_id']
    # image_part_cols = ['item_id', 'image_name']

    def __init__(self, project: str, token: str, bucket: str, dataset: str):
        self.dlfs = gcsfs.GCSFileSystem(project=project, token=token)
        self.pa_fs = fs.PyFileSystem(fs.FSSpecHandler(self.dlfs))
        self.image_path = os.path.join(bucket, dataset, "image")
        self.images = None

    def add_images(self, images: pd.DataFrame):
        pq.write_to_dataset(pa.Table.from_pandas(images), self.image_path,
                            partition_cols=self.image_part_cols, filesystem=self.dlfs)

    def create_datasources(self):
        if self.images is None:
            self.images = ds.dataset(self.image_path, filesystem=self.pa_fs, partitioning="hive")

    def fetch_image_table(self, iid) -> pd.DataFrame:
        self.create_datasources()
        return self.images.to_table(columns=self.image_proj, filter=ds.field("item_id") == iid).to_pandas()

    def fetch_images(self, iid: int) -> Tuple[List[Image.Image], List[str]]:
        """
        Perform a generic flight action
        :param iid item id of images to fetch
        :return: List of images, List of image names
        """
        dl_photos = self.fetch_image_table(iid)
        photos = []
        for photo in dl_photos['crop_data']:
            stream_image = io.BytesIO(photo)
            photos.append(Image.open(stream_image))
        return photos, list(dl_photos['image_name'].values)

    def unique_items(self) -> List[str]:
        self.create_datasources()
        item_ids = self.images.scanner(columns=["item_id"]).to_table()
        return pac.unique(item_ids['item_id']).to_numpy()

    def iid_sampler(self, p: float = 0.2) -> Tuple[List[int], List[int]]:
        iids = list(self.unique_items())
        random.shuffle(iids)
        n = int(p*len(iids))
        return iids[n:], iids[:n]

    def db_query_sampler(self, p: float = 0.2) -> Tuple[List[int], List[int]]:
        iids = list(self.unique_items())
        random.shuffle(iids)
        n = int(p*len(iids))
        return iids[n:], iids[:n]

    def write_images(self, iids: List[int], img_dir: str) -> None:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        count = 0
        for iid in iids:
            d_name = os.path.join(img_dir, str(iid))
            if not os.path.exists(d_name):
                os.makedirs(d_name)
            image_df = self.images.to_table(filter=ds.field("item_id") == iid).to_pandas()
            for _, row in image_df.iterrows():
                img = Image.open(io.BytesIO(row.crop_data))
                iname = str(row.image_name) + ".jpeg"
                fn = os.path.join(img_dir, str(iid), iname)
                img.save(fn)
            if count % 10:
                print(".", end="")
            count += 1


class DatasetN1CropsMulti():

    part_cols = ['item_id', 'capture_uuid']

    # Schema of the N1 Dataset
    n1_crops_multi_schema = pa.schema([
            pa.field('item_id', pa.int64()),
            pa.field('image_name', pa.string()),
            pa.field('capture_uuid', pa.string()),
            pa.field('frame_id', pa.int64()),
            pa.field('crop', pa.binary()),
            pa.field('bow', pa.list_(pa.string())),
            pa.field('dims', pa.list_(pa.float32())),
            pa.field('in_fence', pa.bool_()),
            pa.field('instance_id', pa.int64()),
            pa.field('created_date', pa.timestamp('us', tz='UTC')),
        ],
        metadata={
            "item_id": "Catagorical value of catagorical variable spd_item_id",
            "image_name": "The unique image name, as assigned in the capture",
            "capture_uuid": "UUID-style identifier given to the capture",
            "frame_id": "Index of the image within the capture",
            "crop": "Bytestring of the JPEG image",
            "bow": "Bag of Words text vector from OCR",
            "dims": "The minor and major (resp.) dimensions spanning the contour",
            "instance_id": "Index of the crop within the image, whem multiple are present",
            "created_date": "Date when this image was captured",
        }
    )

    def __init__(self, project: str, token: str, bucket: str):
        fsspec.asyn.iothread[0] = None
        fsspec.asyn.loop[0] = None
        self.bucket = bucket
        self.dlfs = gcsfs.GCSFileSystem(project=project, token=token)
        self.pa_fs = fs.PyFileSystem(fs.FSSpecHandler(self.dlfs))
        self.n1_crops_multi_path = os.path.join(self.bucket, 'n1_crops_multi')
        self.n1_crops_multi_ds = None
        self.chunk_size = 10

    def get_n1_crops_multi(self):
        if self.n1_crops_multi_ds is None:
            self.n1_crops_multi_ds = ds.dataset(self.n1_crops_multi_path,  filesystem=self.pa_fs,
                                     schema=self.n1_crops_multi_schema, partitioning="hive")
        return self.n1_crops_multi_ds

    def add_n1_crops_multi(self, image_tbl: pa.Table) -> None:
        pq.write_to_dataset(image_tbl, self.n1_crops_multi_path, schema=self.n1_crops_multi_schema,
                            partition_cols=self.part_cols, filesystem=self.dlfs)

    def fetch_partitions(self) -> pd.DataFrame:
        data_set = self.get_n1_crops_multi()
        item_pat = re.compile(".*item_id\=([0-9]+)\/capture_uuid=([A-Za-z0-9-]+)/.*")
        item_ids = []
        capture_uuids = []
        for row in data_set.files:
            m = item_pat.search(row)
            item_ids.append(int(m.group(1)))
            capture_uuids.append(m.group(2))
        df = pd.DataFrame({
            'item_id': item_ids,
            'capture_uuid': capture_uuids
        })
        return df.drop_duplicates()
    
    def fetch_training_data(self, iid: int) -> Tuple[List[Image.Image], List[str]]:
        """
        Perform a generic flight action
        :param iid item id of images to fetch
        :return: List of images, List of image names
        """
        dl_data = self.get_n1_crops_multi().to_table(
            filter=ds.field("item_id") == iid).to_pandas()
        photos = []
        bows = dl_data.bow.to_list()
        dims = dl_data.dims.to_list()
        for photo in dl_data['crop']:
            stream_image = io.BytesIO(photo)
            photos.append(Image.open(stream_image))
        return photos, bows, dims
