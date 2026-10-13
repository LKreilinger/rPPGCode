import os
import shutil


def merge_datasets(output_path, dataset_names, merged_name):
    # Copy the splits of several preprocessed datasets into one dataset and join their annotations.txt
    merged_path = os.path.join(output_path, merged_name)
    if os.path.exists(merged_path):
        raise SystemExit("Dataset already exists: " + merged_path)
    for dataset_name in dataset_names:
        dataset_path = os.path.join(output_path, dataset_name)
        if not os.path.isdir(dataset_path):
            raise SystemExit("Dataset not found: " + dataset_path)

    for dataset_name in dataset_names:
        dataset_path = os.path.join(output_path, dataset_name)
        for split in os.listdir(dataset_path):
            split_path = os.path.join(dataset_path, split)
            annotation_file = os.path.join(split_path, "annotations.txt")
            if not os.path.isfile(annotation_file):
                continue
            merged_split_path = os.path.join(merged_path, split)
            os.makedirs(merged_split_path, exist_ok=True)
            for video in os.listdir(split_path):
                video_path = os.path.join(split_path, video)
                if os.path.isdir(video_path):
                    shutil.copytree(video_path, os.path.join(merged_split_path, video))
            with open(annotation_file, "r") as f_r, open(os.path.join(merged_split_path, "annotations.txt"), "a") as f_w:
                f_w.write(f_r.read())
        print("Copied", dataset_name, "->", merged_path)
