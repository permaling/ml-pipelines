import io
import json
import os
import re
import time
from copy import deepcopy
from datetime import datetime
from itertools import chain
from typing import List, Tuple, Dict
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
from PIL import ExifTags, Image, UnidentifiedImageError

from MLPipelines.COCOUtils import rbb_coco_from_seg


class CvatApi:
    TASK_ID = 0
    JOB_ID = 1

    JPEG_PAT = re.compile("\\.[Jj][Pp][Ee]?[Gg](?:\\.[Jj][Pp][Ee]?[Gg])?")
    IMG_EXT = ".jpeg"

    def __init__(self, base_url: str, headers: dict, default_timeout: int = 5):
        self.base_url = base_url
        self.headers = headers
        self.default_timeout = default_timeout

        if self.base_url.endswith("/"):
            self.base_url = self.base_url.rstrip("/")

    def perform_get(self, url: str):
        if not url.startswith(self.base_url):
            if not url.startswith("/"):
                url = f"/{url}"

            url = self.base_url + url

        response = requests.get(url, headers=self.headers, timeout=self.default_timeout)

        if response.status_code == 200:
            return json.loads(response.text)
        raise requests.HTTPError(f"Expected status code 200. Got {response.status_code} instead.")

    def perform_post(self, url: str, data: dict, files: dict = None):
        if not url.startswith(self.base_url):
            if not url.startswith("/"):
                url = f"/{url}"

            url = self.base_url + url

        response = requests.post(url, data=data, files=files, headers=self.headers, timeout=self.default_timeout)

        if response.status_code in [201, 202]:
            return json.loads(response.text)
        raise requests.HTTPError(f"Expected status code 201. Got {response.status_code} instead.")

    def fetch_projects(self) -> Dict:
        return self.perform_get("/projects")

    def fetch_project(self, proj_id: int) -> Tuple[Dict, List, Dict]:
        dat = self.perform_get(f"/projects/{proj_id}")
        labels = {}
        attributes = {}
        job_ids = []
        label_dat = self.perform_get(dat["labels"]["url"])

        for label in label_dat["results"]:
            labels[label["id"]] = label["name"].lower()
            for attr in label["attributes"]:
                attributes[label["id"]] = {attr["id"]: attr["name"]}

        tasks = []
        project_tasks = self.perform_get(dat["tasks"]["url"])
        tasks.extend(project_tasks["results"])

        page_num = 1
        print(f"Processing page {page_num}")

        while project_tasks["next"]:
            page_num += 1
            print(f"Processing page {page_num}")
            project_tasks = self.perform_get(project_tasks["next"])
            tasks.extend(project_tasks["results"])

        print()

        for task in tasks:
            jobs = self.perform_get(task["jobs"]["url"])
            for job in jobs["results"]:
                job_ids.append((job["task_id"], job["id"]))

        return labels, job_ids, attributes

    def fetch_task_meta(self, task_id: int) -> Dict:
        return self.perform_get(f"/tasks/{task_id}/data/meta")

    def fetch_annotations(self, job_id: int) -> Tuple[List, Dict, List]:
        dat = self.perform_get(f"/jobs/{job_id}")
        dat = self.perform_get(f"/tasks/{dat['task_id']}/data/meta")
        images = {}

        for frame_id, frame in enumerate(dat["frames"]):
            frame_name = os.path.split(frame["name"])[-1]
            frame_name = os.path.splitext(frame_name)[0] + CvatApi.IMG_EXT
            images[frame_id] = frame_name

        dat = self.perform_get(f"/jobs/{job_id}/annotations")
        tags = []
        shapes = []

        for tag in dat["tags"]:
            tags.append((tag["label_id"], tag["frame"]))

        for shape in dat["shapes"]:
            shapes.append((shape["frame"], shape["label_id"], shape["type"], shape["points"], shape["attributes"]))

        return tags, images, shapes

    def fetch_track_annotations(self, job_id: int) -> Tuple[List, Dict, List]:
        dat = self.perform_get(f"/jobs/{job_id}")
        dat = self.perform_get(f"/tasks/{dat['task_id']}/data/meta")
        images = {}

        for frame_id, frame in enumerate(dat["frames"]):
            frame_name = os.path.split(frame["name"])[-1]
            frame_name = os.path.splitext(frame_name)[0] + CvatApi.IMG_EXT
            images[frame_id] = frame_name

        dat = self.perform_get(f"/jobs/{job_id}/annotations")
        tags = []
        shapes = []

        for tag in dat["tags"]:
            tags.append((tag["label_id"], tag["frame"]))

        for track in dat["tracks"]:
            track_id = track["id"]
            label_id = track["label_id"]
            end_frame = sorted(track["shapes"], key=lambda shape: shape["frame"])[-1]["frame"]

            for shape in self.interpolate_shapes(track, end_frame):
                shapes.append((shape["frame"], label_id, shape["type"], shape["points"], shape["attributes"], track_id))

        return tags, images, shapes
    
    def fetch_job_images(self, job_id: int, output_dir: str, query: dict = None) -> None:
        def try_get(url: str):
            try:
                return requests.get(url, headers=self.headers, timeout=self.default_timeout)
            except requests.exceptions.ConnectionError:
                print("Connection error. Retrying...")
                return None
        
        # This is awful but it keeps timing out halfway through
        def perform_cyclic_get(url: str, output_dir: str):
            if not url.startswith(self.base_url):
                if not url.startswith("/"):
                    url = f"/{url}"

                url = self.base_url + url

            if not os.path.isdir(output_dir):
                raise FileNotFoundError(f"Directory {output_dir} does not exist.")
                
            output_file = os.path.join(output_dir, "tmp.zip")

            print("Exporting images...")
            response = try_get(url)

            while not response or response.status_code == 202:
                time.sleep(3)
                response = try_get(url)

            if response.status_code == 201:
                print("Downloading images...")

                while not response or response.status_code != 200:
                    time.sleep(3)
                    response = try_get(url + "&action=download")

                with open(output_file, "wb") as file:
                    file.write(response.content)

                with ZipFile(output_file, "r") as zip:
                    for file in zip.namelist():
                        if file.lower().startswith("images/"):
                            zip.getinfo(file).filename = file[len("images/"):]
                            zip.extract(file, os.path.join(output_dir, str(job_id)))

                os.remove(output_file)
                print("Images downloaded.")
            else:
                raise requests.HTTPError(f"Expected status code 201. Got {response.status_code} instead.")

        additional_queries = "&".join([f"{k}={v}" for k, v in query.items()])
        url = f"/jobs/{job_id}/dataset?use_default_location=false&location=local&format=CVAT+for+images+1.1{'&' + additional_queries if query else ''}"
        perform_cyclic_get(url, output_dir)

    # TODO: Test creating tasks and adding images with the new API, these methods may not work as they are

    # def create_cvat_task(self, project_id: int, task_name: str) -> int:
    #     data = {"name": task_name, "project_id": project_id}
    #     return self.perform_post("/tasks", data)["id"]

    # def add_images_to_task(self, task_id: int, images: list):
    #     data = {"image_quality": 70}
    #     files = {}

    #     for i, image_file in enumerate(images):
    #         try:
    #             if Image.open(image_file).format.lower() in ["png", "jpg", "jpeg"]:
    #                 files[f"client_files[{i}]"] = open(image_file, "rb")
    #         except UnidentifiedImageError as e:
    #             print(f"UnidentifiedImageError: {e}")

    #     if files:
    #         self.perform_post(f"/tasks/{task_id}/data", data, files)

    @staticmethod
    def load_image_to_bytes(fn: str):
        image = Image.open(fn)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif:
            if exif.get(orientation) == 3:
                image = image.rotate(180, expand=True)
            elif exif.get(orientation) == 6:
                image = image.rotate(270, expand=True)
            elif exif.get(orientation) == 8:
                image = image.rotate(90, expand=True)
        with io.BytesIO() as image_bytes:
            image.save(image_bytes, format='JPEG')
            return image_bytes.getvalue()
    
    @staticmethod
    def lookup_tags(frame_id: int, tags: list, labels: dict) -> List:
        image_tags = []
        for tag in tags:
            if tag[1] == frame_id:
                t = tag[0]
                image_tags.append(labels[t])
        return image_tags

    @staticmethod
    def create_image_feather(
        labels: dict,
        images: dict,
        tags: list,
        project_id: int,
        cvat_ids: tuple,
        image_dir: str,
        drop_prefix: bool = False,
        respect_exif: bool = True,
    ) -> pd.DataFrame:
        image_bytes = []
        image_name = []
        image_tags = []

        for frame_id in images:
            frame_name = images[frame_id]

            # Try to some dataset inconsistencies
            if drop_prefix:
                # Soren: this was from Dani - need to investigate
                if len(frame_name.split("_")) >= 2:  # Make sure that there is actually a prefix to drop...
                    # NOTE: SOREN: I removed an unchecked exception here; if you get
                    # an error it would be acceptable to add it back in but as a "checked exception"
                    if not frame_name.split("_")[0] == "image":  # Might need to drop "image" as a prefix
                        int(frame_name.split("_")[0])  # Check to see if prefix is a task id
                    # Remove the task id from first position in the name
                    frame_name = "_".join(frame_name.split("_")[1:])
                    # Remove redundant JPEG prefixes
                    frame_name = CvatApi.JPEG_PAT.sub("", frame_name)
                    frame_name = frame_name + CvatApi.IMG_EXT

            # Load versions of CVAT doesn't respect EXIF, thus we need this flag to say if
            # if the image needs to be rotated in load_image_to_bytes based on EXIF information
            img_fn = os.path.join(image_dir, frame_name)

            try:
                if respect_exif:
                    image_bytes.append(load_image_to_bytes(img_fn))
                else:
                    image = Image.open(img_fn)
                    with io.BytesIO() as ib:
                        image.save(ib, format="JPEG")
                        image_bytes.append(ib.getvalue())

                # Lowercase labels and remove file extensions
                im = os.path.splitext(frame_name)[0]
                image_name.append(im)
                image_tags.append(CvatApi.lookup_tags(frame_id, tags, labels))

            except FileNotFoundError as e:
                print(f"FileNotFoundError: {img_fn} {e}")

        df = pd.DataFrame(
            {
                "project_id": project_id,
                "task_id": cvat_ids[CvatApi.TASK_ID],
                "job_id": cvat_ids[CvatApi.JOB_ID],
                "image_name": image_name,
                "image_bytes": image_bytes,
                "tags": image_tags,
                "ts": datetime.now()
            }
        )

        return df

    @staticmethod
    def rect_to_polygon(points: list) -> List:
        left = points[0]
        top = points[1]
        right = points[2]
        bottom = points[3]

        # make a polygon TL, TR, BR, BL, TL
        return [left, top, right, top, right, bottom, left, bottom, left, top]

    @staticmethod
    def create_anno_feather(
        labels: dict,
        attr_labs: dict,
        images: dict,
        shapes: list,
        project_id: int,
        cvat_ids: tuple,
        drop_prefix: bool = False,
    ) -> pd.DataFrame:
        category = []
        image_name = []
        segmentation = []
        rcoco = []
        boxes = []
        gt_attrs = []
        track_ids = []

        for shape in shapes:
            frame_id = shape[0]
            label_id = shape[1]
            shape_type = shape[2]
            polygon = shape[3]
            attributes = shape[4]

            if len(shape) > 5:
                track_id = shape[5]
            else:
                track_id = -1

            if shape_type == "rectangle":
                polygon = CvatApi.rect_to_polygon(polygon)

            if len(polygon) < 5:
                print(f"Found to few points in polygon in {images[frame_id]} job f{cvat_ids}, skipping...")
                continue

            try:
                coco, box = rbb_coco_from_seg(polygon)
            except Exception as err:
                print(f"Found degenerate polygon in in {images[frame_id]} job f{cvat_ids}, skipping...")
                print(err)
                continue

            category.append(labels[label_id])
            im = os.path.splitext(images[frame_id])[0]

            if drop_prefix and len(im.split("_")) >= 2:
                # Remove the task id from first position in the name
                im = "_".join(im.split("_")[1:])

            im1 = CvatApi.JPEG_PAT.sub("", im)

            if im1 != im:
                print("Removed extension", im, "to", im1)
                im = im1

            image_name.append(im)
            segmentation.append(polygon)
            rcoco.append(list(coco))
            boxes.append(list(box))
            gt = []

            # Get valid attributes for this label
            if label_id in attr_labs:
                attr_types = attr_labs[label_id]

                for attr in attributes:
                    # Get the ground truth item_id
                    attr_type = attr_types.get(attr["spec_id"])
                    val = ' "' + attr["value"] + '"}'

                    if attr_type == "Item ID":
                        gt.append('{"iid":"' + val)
                    elif attr_type == "UUID":
                        gt.append('{"uuid":' + val)
                    elif attr_type == "Text":
                        gt.append('{"text":' + val)

            gt_attrs.append("[" + ", ".join(gt) + "]")
            track_ids.append(track_id)

        df = pd.DataFrame(
            {
                "project_id": project_id,
                "task_id": cvat_ids[CvatApi.TASK_ID],
                "job_id": cvat_ids[CvatApi.JOB_ID],
                "track_id": track_ids,
                "image_name": image_name,
                "category": category,
                "segmentation": segmentation,
                "rcoco": rcoco,
                "coco": boxes,
                "gt_attr": gt_attrs,
                "ts": datetime.now()
            }
        )

        return df

    def interpolate_shapes(self, track, end_frame):
        # Taken and modified from https://github.com/opencv/cvat/blob/develop/cvat/apps/dataset_manager/annotation.py (MIT License)

        def deepcopy_simple(v):
            # Default deepcopy is very slow

            if isinstance(v, dict):
                return {k: deepcopy_simple(vv) for k, vv in v.items()}
            elif isinstance(v, (list, tuple, set)):
                return type(v)(deepcopy_simple(vv) for vv in v)
            elif isinstance(v, (int, float, str, bool)) or v is None:
                return v
            else:
                return deepcopy(v)

        def copy_shape(source, frame, points=None, rotation=None):
            copied = source.copy()
            copied["attributes"] = deepcopy_simple(source["attributes"])

            copied["keyframe"] = False
            copied["frame"] = frame
            if rotation is not None:
                copied["rotation"] = rotation

            if points is None:
                points = copied["points"]

            if isinstance(points, np.ndarray):
                points = points.tolist()
            else:
                points = points.copy()

            if points is not None:
                copied["points"] = points

            return copied

        def interpolate_position(left_position, right_position, offset):
            def to_array(points):
                return np.asarray(list(map(lambda point: [point["x"], point["y"]], points))).flatten()

            def to_points(array):
                return list(map(lambda point: {"x": point[0], "y": point[1]}, np.asarray(array).reshape(-1, 2)))

            def curve_length(points):
                length = 0
                for i in range(1, len(points)):
                    dx = points[i]["x"] - points[i - 1]["x"]
                    dy = points[i]["y"] - points[i - 1]["y"]
                    length += np.sqrt(dx**2 + dy**2)
                return length

            def curve_to_offset_vec(points, length):
                offset_vector = [0]
                accumulated_length = 0
                for i in range(1, len(points)):
                    dx = points[i]["x"] - points[i - 1]["x"]
                    dy = points[i]["y"] - points[i - 1]["y"]
                    accumulated_length += np.sqrt(dx**2 + dy**2)
                    offset_vector.append(accumulated_length / length)

                return offset_vector

            def find_nearest_pair(value, curve):
                minimum = [0, abs(value - curve[0])]
                for i in range(1, len(curve)):
                    distance = abs(value - curve[i])
                    if distance < minimum[1]:
                        minimum = [i, distance]

                return minimum[0]

            def match_left_right(left_curve, right_curve):
                matching = {}
                for i, left_curve_item in enumerate(left_curve):
                    matching[i] = [find_nearest_pair(left_curve_item, right_curve)]
                return matching

            def match_right_left(left_curve, right_curve, left_right_matching):
                matched_right_points = list(chain.from_iterable(left_right_matching.values()))
                unmatched_right_points = filter(lambda x: x not in matched_right_points, range(len(right_curve)))
                updated_matching = deepcopy_simple(left_right_matching)

                for right_point in unmatched_right_points:
                    left_point = find_nearest_pair(right_curve[right_point], left_curve)
                    updated_matching[left_point].append(right_point)

                for key, value in updated_matching.items():
                    updated_matching[key] = sorted(value)

                return updated_matching

            def reduce_interpolation(interpolated_points, matching, left_points, right_points):
                def average_point(points):
                    sum_x = 0
                    sum_y = 0
                    for point in points:
                        sum_x += point["x"]
                        sum_y += point["y"]

                    return {"x": sum_x / len(points), "y": sum_y / len(points)}

                def compute_distance(point1, point2):
                    return np.sqrt((point1["x"] - point2["x"]) ** 2 + ((point1["y"] - point2["y"]) ** 2))

                def minimize_segment(base_length, n, start_interpolated, stop_interpolated):
                    threshold = base_length / (2 * n)
                    minimized = [interpolated_points[start_interpolated]]
                    latest_pushed = start_interpolated
                    for i in range(start_interpolated + 1, stop_interpolated):
                        distance = compute_distance(interpolated_points[latest_pushed], interpolated_points[i])

                        if distance >= threshold:
                            minimized.append(interpolated_points[i])
                            latest_pushed = i

                    minimized.append(interpolated_points[stop_interpolated])

                    if len(minimized) == 2:
                        distance = compute_distance(
                            interpolated_points[start_interpolated], interpolated_points[stop_interpolated]
                        )

                        if distance < threshold:
                            return [average_point(minimized)]

                    return minimized

                reduced = []
                interpolated_indexes = {}
                accumulated = 0
                for i in range(len(left_points)):
                    interpolated_indexes[i] = []
                    for _ in range(len(matching[i])):
                        interpolated_indexes[i].append(accumulated)
                        accumulated += 1

                def left_segment(start, stop):
                    start_interpolated = interpolated_indexes[start][0]
                    stop_interpolated = interpolated_indexes[stop][0]

                    if start_interpolated == stop_interpolated:
                        reduced.append(interpolated_points[start_interpolated])
                        return

                    base_length = curve_length(left_points[start : stop + 1])
                    n = stop - start + 1

                    reduced.extend(minimize_segment(base_length, n, start_interpolated, stop_interpolated))

                def right_segment(left_point):
                    start = matching[left_point][0]
                    stop = matching[left_point][-1]
                    start_interpolated = interpolated_indexes[left_point][0]
                    stop_interpolated = interpolated_indexes[left_point][-1]
                    base_length = curve_length(right_points[start : stop + 1])
                    n = stop - start + 1

                    reduced.extend(minimize_segment(base_length, n, start_interpolated, stop_interpolated))

                previous_opened = None
                for i in range(len(left_points)):
                    if len(matching[i]) == 1:
                        if previous_opened is not None:
                            if matching[i][0] == matching[previous_opened][0]:
                                continue
                            else:
                                start = previous_opened
                                stop = i - 1
                                left_segment(start, stop)
                                previous_opened = i
                        else:
                            previous_opened = i
                    else:
                        if previous_opened is not None:
                            start = previous_opened
                            stop = i - 1
                            left_segment(start, stop)
                            previous_opened = None

                        right_segment(i)

                if previous_opened is not None:
                    left_segment(previous_opened, len(left_points) - 1)

                return reduced

            left_points = to_points(left_position["points"])
            right_points = to_points(right_position["points"])
            left_offset_vec = curve_to_offset_vec(left_points, curve_length(left_points))
            right_offset_vec = curve_to_offset_vec(right_points, curve_length(right_points))

            matching = match_left_right(left_offset_vec, right_offset_vec)
            completed_matching = match_right_left(left_offset_vec, right_offset_vec, matching)

            interpolated_points = []
            for left_point_index, left_point in enumerate(left_points):
                for right_point_index in completed_matching[left_point_index]:
                    right_point = right_points[right_point_index]
                    interpolated_points.append(
                        {
                            "x": left_point["x"] + (right_point["x"] - left_point["x"]) * offset,
                            "y": left_point["y"] + (right_point["y"] - left_point["y"]) * offset,
                        }
                    )

            reduced_points = reduce_interpolation(interpolated_points, completed_matching, left_points, right_points)

            return to_array(reduced_points).tolist()

        def interpolate(shape0, shape1):
            if not shape0["type"] == shape1["type"] == "polygon":
                raise NotImplementedError()

            shapes = []

            # Make the polygon closed for computations
            shape0 = shape0.copy()
            shape1 = shape1.copy()
            shape0["points"] = shape0["points"] + shape0["points"][:2]
            shape1["points"] = shape1["points"] + shape1["points"][:2]

            distance = shape1["frame"] - shape0["frame"]
            for frame in range(shape0["frame"] + 1, shape1["frame"]):
                offset = (frame - shape0["frame"]) / distance
                points = interpolate_position(shape0, shape1, offset)

                shapes.append(copy_shape(shape0, frame, points))

            # Remove the extra point added
            shape0["points"] = shape0["points"][:-2]
            shape1["points"] = shape1["points"][:-2]
            for shape in shapes:
                shape["points"] = shape["points"][:-2]

            return shapes

        def propagate(shape, end_frame):
            return [copy_shape(shape, i) for i in range(shape["frame"] + 1, end_frame)]

        shapes = []
        prev_shape = None

        for shape in sorted(track["shapes"], key=lambda shape: shape["frame"]):
            curr_frame = shape["frame"]

            if prev_shape and end_frame <= curr_frame:
                # If we exceed the end_frame and there was a previous shape,
                # we still need to interpolate up to the next keyframe,
                # but keep the results only up to the end_frame:
                #        vvvvvvv
                # ---- | ------- | ----- | ----->
                #     prev      end   cur kf
                interpolated = interpolate(prev_shape, shape)
                interpolated.append(shape)

                for shape in sorted(interpolated, key=lambda shape: shape["frame"]):
                    if shape["frame"] < end_frame:
                        shapes.append(shape)
                    else:
                        break

                # Update the last added shape
                shape["keyframe"] = True
                prev_shape = shape

                break  # The track finishes here

            if prev_shape:
                if curr_frame > prev_shape["frame"]:  # Catch invalid tracks
                    # print("ERROR: invalid track. Continuing...")
                    pass

                # Propagate attributes
                for attr in prev_shape["attributes"]:
                    if attr["spec_id"] not in map(lambda el: el["spec_id"], shape["attributes"]):
                        shape["attributes"].append(deepcopy_simple(attr))

                if not prev_shape["outside"]:
                    shapes.extend(interpolate(prev_shape, shape))

            shape["keyframe"] = True
            shapes.append(shape)
            prev_shape = shape

        if prev_shape and not prev_shape["outside"]:
            # When the latest keyframe of a track is less than the end_frame
            # and it is not outside, need to propagate
            shapes.extend(propagate(prev_shape, end_frame))

        shapes = [
            shape
            for shape in shapes
            # After interpolation there can be a finishing frame
            # outside of the task boundaries. Filter it out to avoid errors.
            # https://github.com/openvinotoolkit/cvat/issues/2827
            if track["frame"] <= shape["frame"] < end_frame
            # Exclude outside shapes.
            # Keyframes should be included regardless the outside value
            # If really needed, they can be excluded on the later stages,
            # but here they represent a finishing shape in a visible sequence
            if shape["keyframe"] or not shape["outside"]
        ]

        return shapes
