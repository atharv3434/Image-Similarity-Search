"""Build a feature index over every image in the database.

Usage:
    python src/build_index.py [--config config.yaml]

Extracts a feature vector for every image in `images_dir` and saves them,
along with filenames and metadata, to `index_path` for fast repeated
similarity search without recomputing features each time.

"""

import argparse
import os
import sys

import cv2
import joblib
import numpy as np

sys.path.append(os.path.dirname(__file__))
from utils import load_config, load_metadata
from features import extract_features


def build_index(config):
    metadata = load_metadata(config["metadata_path"])
    records = metadata["images"]

    vectors = []
    kept_records = []
    for record in records:
        path = os.path.join(config["images_dir"], record["file"])
        image = cv2.imread(path)
        if image is None:
            print(f"  Warning: could not read {path}, skipping.")
            continue
        vec = extract_features(image, config)
        vectors.append(vec)
        kept_records.append(record)

    feature_matrix = np.vstack(vectors)
    return feature_matrix, kept_records


def main():
    parser = argparse.ArgumentParser(description="Build the image feature index.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Extracting features for images in {config['images_dir']} ...")
    feature_matrix, records = build_index(config)
    print(f"Built index: {feature_matrix.shape[0]} images, {feature_matrix.shape[1]}-dim feature vectors.")

    os.makedirs(os.path.dirname(config["index_path"]), exist_ok=True)
    joblib.dump({
        "feature_matrix": feature_matrix,
        "records": records,
        "config": config,
    }, config["index_path"])
    print(f"Index saved to {config['index_path']}")


if __name__ == "__main__":
    main()
