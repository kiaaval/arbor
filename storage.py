import csv
import json


def fit(tree, label: str, output: str):
    with open(output, "w", encoding="utf-8") as file:
        data = {"label": label, "tree": tree}
        json.dump(data, file, indent=2)
    print("Model created at " + output)


def predict(data, model):
    with open(model, encoding="utf-8") as f:
        saved = json.load(f)
    tree = saved["tree"]
    label = saved["label"]

    with open(data, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return tree, rows, label


def write_predictions(rows, predictions, output, label):
    column = "prediction " + label
    fieldnames = list(rows[0].keys()) + [column]
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, prediction in zip(rows, predictions):
            writer.writerow({**row, column: prediction})
    print("Predictions saved to " + output)
