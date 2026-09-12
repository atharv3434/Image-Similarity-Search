# Image Similarity Search (Feature Extraction, Classical CV)

A content-based image retrieval project: extracts hand-engineered visual
features from images and finds the most similar images in a database via
nearest-neighbor search — no neural network or pretrained weights required.

Combines two complementary, classical feature types:
- **Color histogram (HSV)** — what colors are present
- **HOG (Histogram of Oriented Gradients)** — shape and edge structure

into a single normalized feature vector, then uses scikit-learn's
`NearestNeighbors` for cosine-similarity search.

## Project structure

```
image-similarity-search/
├── config.yaml                  # feature settings, weights, search params
├── requirements.txt
├── data/
│   ├── generate_data.py         # (re)generates the synthetic image database
│   ├── images/                  # pre-generated sample images
│   └── metadata.json            # category labels (for evaluation only)
├── src/
│   ├── utils.py                 # config + metadata loading
│   ├── features.py              # color histogram + HOG extraction
│   ├── build_index.py           # extracts features for the whole database
│   ├── query.py                 # find images similar to a given query image
│   └── evaluate.py              # leave-one-out precision@K evaluation
├── models/
│   └── feature_index.joblib     # saved feature index (built by build_index.py)
├── output/                       # query visualizations + evaluation report
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Build the index

```bash
python src/build_index.py
```

Extracts a combined feature vector for every image in `data/images/` and
saves them all to `models/feature_index.joblib`. Re-run this any time you
add new images or change feature settings in `config.yaml`.

## 2. Search

```bash
python src/query.py --image data/images/star_067.png --top-k 5 --visualize
```

Prints the most similar images by cosine similarity:

```
Top 5 matches for data/images/star_067.png:
  1. star_074.png         category=star       similarity=0.829
  2. square_031.png       category=square     similarity=0.720
  3. star_068.png         category=star       similarity=0.719
  4. triangle_050.png     category=triangle   similarity=0.710
  5. star_073.png         category=star       similarity=0.702
```

With `--visualize`, saves a side-by-side montage (`output/query_<file>.png`)
of the query and its matches. Works on any image path, not just ones already
in the database.

## 3. Evaluate

```bash
python src/evaluate.py
```

Runs a leave-one-out check: for every image, query the index (excluding
itself) and measure what fraction of the retrieved neighbors share its
shape category. Reports precision@K for several K values, a per-category
breakdown, and saves example montages + a full report to
`output/evaluation_report.md`.

Typical result on the bundled data:

```
Precision@1: 0.625
Precision@3: 0.525
Precision@5: 0.458

Per-category precision@3:
  circle     0.600
  square     0.417
  star       0.800
  triangle   0.283
```

Notice precision drops as K grows (harder to keep finding same-category
matches further down the ranking), and categories vary in difficulty —
stars have a distinctive silhouette that HOG captures well, while triangles
under rotation are easily confused with other shapes.

## Why color sometimes "wins" over shape

Because color histogram and HOG are combined with fixed weights
(`color_weight` / `hog_weight` in `config.yaml`, default 0.5/0.5), a strong
color match can sometimes outrank a same-shape match with different color —
e.g. a green circle may retrieve a green square before a red circle. This is
a genuine, instructive property of hand-engineered multi-feature search, not
a bug: adjust the weights (or set one to 0) to see retrieval quality shift
between "color-driven" and "shape-driven" behavior.

## Using your own images

1. Point `images_dir` in `config.yaml` at your own image folder.
2. If you want evaluation metrics, create a `metadata.json` in the same
   structure as the sample one, with a `category` label per image
   (otherwise, just skip `evaluate.py` and use `build_index.py` + `query.py`
   directly — no labels needed for search itself).
3. Re-run `python src/build_index.py`.

## Extending this project

- **More feature types**: add a function to `features.py` (e.g. local
  binary patterns for texture, or color moments) and include it in
  `extract_features()`.
- **Faster search at scale**: swap `NearestNeighbors` (brute-force) for an
  approximate method like FAISS or an LSH index once the database grows
  beyond a few thousand images.
- **Learned features**: for real-world photos (not clean synthetic shapes),
  embeddings from a pretrained CNN typically outperform hand-engineered
  features — this project is a solid classical baseline to compare against.
