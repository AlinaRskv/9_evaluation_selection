from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def create_pipeline(
    use_scaler: bool, scaler: str,  k: int, weights: str, p: int, model: str, criterion: str, max_depth: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler and scaler == 'StandardScaler':
        pipeline_steps.append(("scaler", StandardScaler()))
    elif use_scaler and scaler == 'MinMaxScaler':
        pipeline_steps.append(("scaler", MinMaxScaler()))
    if model == 'KNeighbors':
        pipeline_steps.append(
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=k,
                    weights=weights,
                    p=p
                ),
            )
        )
    elif model == 'DecisionTree':
        pipeline_steps.append(
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=42,
                    criterion=criterion,
                    max_depth=max_depth
                ),
            )
        )
    return Pipeline(steps=pipeline_steps)