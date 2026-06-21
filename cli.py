import click

from dataset import load_data
from storage import fit as model_fit
from storage import predict as load_predict_data
from storage import write_predictions
from tree import model
from tree import predict as predict_row


@click.group()
def cli():
    pass


@cli.command()
@click.option("--data", required=True, help="Path to training CSV")
@click.option("--target", required=True, help="Target column name")
@click.option(
    "--features",
    required=False,
    default=None,
    help="Comma-seperated feature columns",
)
@click.option("--output", required=True, help="Path to save model")
def fit(data, target, features, output):
    features = [f.strip() for f in features.split(",")] if features else None
    features, labels = load_data(data, target, features)
    tree = model(features, labels)
    model_fit(tree, target, output)


@cli.command()
@click.option("--data", required=True, help="Path to input CSV")
@click.option("--model", required=True, help="Path to .arbor model file")
@click.option("--output", required=True, help="Path to save predictions CSV")
def predict(data, model, output):
    tree, rows, label = load_predict_data(data, model)
    predictions = [predict_row(tree, row) for row in rows]
    write_predictions(rows, predictions, output, label)


@cli.command()
@click.option("--predictions", required=True, help="Path to predictions CSV")
@click.option("--answers", required=True, help="Path to labeled CSV with ground truth")
@click.option("--target", required=True, help="Target column name in answers CSV")
def evaluate(predictions, answers, target):
    pass


if __name__ == "__main__":
    cli()
