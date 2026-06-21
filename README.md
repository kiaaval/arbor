# arbor

ML models built from scratch in pure Python — no scikit-learn, no numpy, no magic.

Arbor is a from-scratch machine learning toolkit. The goal is to grow it into a collection of models and prediction tools that work on any tabular CSV data, all implemented in plain Python so the algorithms stay readable and hackable.

Right now it ships an **ID3 decision tree** and a CLI for training, predicting, and evaluating.

## Quick start

```bash
# Train a model
python cli.py fit --data train.csv --target Label --output model.arbor

# Predict on new data
python cli.py predict --data test.csv --model model.arbor --output predictions.csv

# NOT AVAIlABLE YET
# Evaluate predictions against ground truth
python cli.py evaluate --predictions predictions.csv --answers labeled.csv --target Label
```

`--features` is optional — if omitted, every column except the target is used as a feature.

## How it works

The current model is an ID3 decision tree:

1. **Entropy** measures how mixed a set of labels is (0 = pure, 1 = even split).
2. **Information gain** picks the feature that reduces entropy the most after splitting.
3. The tree recurses on each branch until every leaf is pure or no features remain (falls back to majority vote).

Models are saved as JSON (`.arbor` files) so you can inspect or version them.

## Project layout

| File | Role |
|---|---|
| `cli.py` | Click CLI — `fit`, `predict`, `evaluate` commands |
| `dataset.py` | Loads CSV data into feature columns + labels |
| `tree.py` | ID3 core — `build_tree()`, `model()`, `predict()` |
| `entropy.py` | `get_entropy()` and `information_gain()` |
| `storage.py` | Model serialization and prediction I/O |
| `schema.py` | TypedDict shapes for internal data structures |

## Roadmap

- Evaluation metrics (accuracy, confusion matrix)
- Additional models beyond ID3
- Continuous feature support
- Train/test splitting utilities
