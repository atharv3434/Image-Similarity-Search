"""Find the most similar images to a query image.

Usage:
    python src/query.py --image data/images/star_067.png [--top-k 5] [--visualize]

Loads the pre-built feature index, extracts features for the query image
the same way the index was built, and returns the nearest neighbors by
cosine similarity.
"""

import argparse
import os
import sys

import cv2
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.dirname(__file__))
from utils import load_config
from features import extract_features


def load_index(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No feature index found at '{index_path}'. Run `python src/build_index.py` first."
        )
    return joblib.load(index_path)


def find_similar(query_vector, index, top_k, exclude_file=None):
    """Return the top_k most similar records to `query_vector`.

    If `exclude_file` matches a record's filename, that record is skipped
    (used for leave-one-out evaluation, so a query never "finds itself").
    """
    matrix = index["feature_matrix"]
    records = index["records"]

    metric = index["config"].get("distance_metric", "cosine")
    n_neighbors = min(top_k + 1, len(records))  # +1 in case we need to drop a self-match
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(matrix)
    distances, indices = nn.kneighbors(query_vector.reshape(1, -1))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        record = records[idx]
        if exclude_file is not None and record["file"] == exclude_file:
            continue
        similarity = 1 - dist if metric == "cosine" else -dist
        results.append({**record, "similarity": round(float(similarity), 4)})
        if len(results) == top_k:
            break

    return results


def build_montage(query_path, results, images_dir, cell_size=128):
    """Lay the query image and its top matches out side by side for visual inspection."""
    query_img = cv2.resize(cv2.imread(query_path), (cell_size, cell_size))
    cv2.putText(query_img, "QUERY", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    tiles = [query_img]
    for r in results:
        img = cv2.resize(cv2.imread(os.path.join(images_dir, r["file"])), (cell_size, cell_size))
        label = f"{r['category']} {r['similarity']:.2f}"
        cv2.putText(img, label, (5, cell_size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(img)

    separator = np.full((cell_size, 4, 3), 0, dtype=np.uint8)
    spaced = []
    for i, t in enumerate(tiles):
        if i > 0:
            spaced.append(separator)
        spaced.append(t)
    return np.hstack(spaced)


def main():
    parser = argparse.ArgumentParser(description="Find similar images to a query.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--image", required=True, help="Path to the query image")
    parser.add_argument("--top-k", type=int, help="Number of results to return (overrides config)")
    parser.add_argument("--visualize", action="store_true", help="Save a montage image of the results")
    args = parser.parse_args()

    config = load_config(args.config)
    top_k = args.top_k or config.get("top_k", 5)

    index = load_index(config["index_path"])

    query_img = cv2.imread(args.image)
    if query_img is None:
        raise FileNotFoundError(f"Could not read image at '{args.image}'")

    query_vector = extract_features(query_img, index["config"])
    query_filename = os.path.basename(args.image)
    results = find_similar(query_vector, index, top_k, exclude_file=query_filename)

    print(f"Top {len(results)} matches for {args.image}:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['file']:20s} category={r['category']:10s} similarity={r['similarity']:.3f}")

    if args.visualize:
        os.makedirs(config["output_dir"], exist_ok=True)
        montage = build_montage(args.image, results, config["images_dir"])
        out_path = os.path.join(config["output_dir"], f"query_{query_filename}")
        cv2.imwrite(out_path, montage)
        print(f"\nSaved visualization to {out_path}")


if __name__ == "__main__":
    main()
