# Image Similarity Search — Evaluation Report

Leave-one-out evaluation: for every image, query the index (excluding itself) and check whether the retrieved neighbors share its category.

## Precision@K (overall)

| K | Precision |
|---|---|
| 1 | 0.625 |
| 3 | 0.525 |
| 5 | 0.458 |

## Per-category precision@3

| Category | Precision |
|---|---|
| circle | 0.600 |
| square | 0.417 |
| star | 0.800 |
| triangle | 0.283 |

## Example queries

**Query category: square**
![square_032.png](example_square_032.png)

**Query category: star**
![star_067.png](example_star_067.png)

**Query category: square**
![square_039.png](example_square_039.png)

**Query category: circle**
![circle_013.png](example_circle_013.png)
