"""Generate a small synthetic image database for similarity search practice.

Creates images of simple shapes with varying color, size, and rotation, each
labeled with a category (the shape type) used later purely for evaluating
retrieval quality — the search system itself never sees these labels.

This project ships with pre-generated images already in place
(data/images/ and data/metadata.json), so you don't need to run this to try
the project out. Run it again for a fresh random set.

Usage:
    python data/generate_data.py [--n-per-class 20] [--seed 42]
    
"""

import argparse
import json
import os

import numpy as np
import cv2

SHAPES = ["circle", "square", "triangle", "star"]
CANVAS_SIZE = 128
BACKGROUND_COLOR = (245, 245, 245)  # light gray, BGR

# A palette of distinct colors so color histograms have something to key on.
PALETTE = {
    "red": (50, 50, 210),
    "green": (60, 170, 60),
    "blue": (200, 100, 40),
    "yellow": (40, 200, 220),
    "purple": (180, 60, 170),
    "orange": (30, 130, 230),
}


def _draw_star(canvas, center, outer_r, color):
    cx, cy = center
    pts = []
    for i in range(10):
        angle = np.pi / 5 * i - np.pi / 2
        r = outer_r if i % 2 == 0 else outer_r * 0.45
        pts.append([cx + r * np.cos(angle), cy + r * np.sin(angle)])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(canvas, [pts], color)


def _draw_shape(canvas, shape, center, size, color, rotation_deg):
    cx, cy = center
    if shape == "circle":
        cv2.circle(canvas, (cx, cy), size // 2, color, thickness=-1)
    elif shape == "square":
        half = size // 2
        pts = np.array([
            [-half, -half], [half, -half], [half, half], [-half, half]
        ], dtype=np.float32)
        theta = np.radians(rotation_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pts = (pts @ rot.T) + np.array([cx, cy])
        cv2.fillPoly(canvas, [pts.astype(np.int32)], color)
    elif shape == "triangle":
        half = size // 2
        pts = np.array([[0, -half], [-half, half], [half, half]], dtype=np.float32)
        theta = np.radians(rotation_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pts = (pts @ rot.T) + np.array([cx, cy])
        cv2.fillPoly(canvas, [pts.astype(np.int32)], color)
    elif shape == "star":
        _draw_star(canvas, (cx, cy), size // 2, color)


def generate_image(rng, shape):
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), BACKGROUND_COLOR, dtype=np.uint8)
    color_name = rng.choice(list(PALETTE.keys()))
    color = PALETTE[color_name]
    size = int(rng.integers(50, 90))
    rotation = int(rng.integers(0, 360))
    margin = size // 2 + 6
    cx = int(rng.integers(margin, CANVAS_SIZE - margin))
    cy = int(rng.integers(margin, CANVAS_SIZE - margin))

    _draw_shape(canvas, shape, (cx, cy), size, color, rotation)
    return canvas, color_name, size, rotation


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic image database.")
    parser.add_argument("--n-per-class", type=int, default=20, help="Images per shape category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    records = []
    idx = 0
    for shape in SHAPES:
        for _ in range(args.n_per_class):
            canvas, color_name, size, rotation = generate_image(rng, shape)
            filename = f"{shape}_{idx:03d}.png"
            cv2.imwrite(os.path.join(images_dir, filename), canvas)
            records.append({
                "file": filename,
                "category": shape,
                "color": color_name,
                "size": size,
                "rotation": rotation,
            })
            idx += 1

    rng.shuffle(records)  # so ordering doesn't trivially reveal category

    metadata_path = os.path.join(args.out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"categories": SHAPES, "images": records}, f, indent=2)

    print(f"Wrote {len(records)} images to {images_dir}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
