from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from numpy import mean
from numpy import std

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from .data import get_dataset
from .pipeline import create_pipeline

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.3,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)

@click.option(
    "--model",
    default='KNeighbors',
    type=str,
    show_default=True,
)

@click.option(
    "--criterion",
    default='gini',
    type=str,
    show_default=True,
)

@click.option(
    "--max_depth",
    default=None,
    type=int,
    show_default=True,
)

@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)

@click.option(
    "--scaler",
    default='Standard',
    type=str,
    show_default=True,
)

@click.option(
    "--k",
    default=5,
    type=int,
    show_default=True,
)

@click.option(
    "--weights",
    default='uniform',
    type=str,
    show_default=True,
)

@click.option(
    "--p",
    default=2,
    type=int,
    show_default=True,
)

@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    test_split_ratio: float,
    model: str,
    criterion: str,
    max_depth: int,
    use_scaler: bool,
    scaler: str,
    k: int,
    weights: str,
    p: int,
    random_state: int,

) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, scaler, k, weights, p, model, criterion, max_depth)
        pipeline.fit(features_train, target_train)
        accuracy = accuracy_score(target_val, pipeline.predict(features_val))
        roc_auc = roc_auc_score(target_val, pipeline.predict_proba(features_val), multi_class='ovr')
        f1 = f1_score(target_val, pipeline.predict(features_val), average='macro')
        cv = KFold(n_splits=10, random_state=random_state, shuffle=True)
        scores = cross_val_score(pipeline, features_train, target_train, scoring='accuracy', cv=cv, n_jobs=-1)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("scaler type", scaler)
        mlflow.log_param("k", k)
        mlflow.log_param("weights", weights)
        mlflow.log_param("p", p)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc_score", roc_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("K-fold CV mean score", mean(scores))
        mlflow.log_metric("K-fold CV std of scores", std(scores))
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"roc_auc_score: {roc_auc}.")
        click.echo(f"f1_score: {f1}.")
        click.echo(f"K-fold CV mean score {mean(scores)} with standard deviation {std(scores)}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")