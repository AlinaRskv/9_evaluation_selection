Homework for RS School Machine Learning course.
This project uses the Forest train dataset.
This package allows you to train model to predict an integer classification for the forest cover type.

1. Clone this repository to your machine.

2. Download [Forest Train Dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction), save csv locally (default path is data/heart.csv in repository's root).

3. Make sure Python 3.9.12 and Poetry are installed on your machine.

4. Install the project dependencies (run this and following commands in a terminal, from the root of a cloned repository):

```
poetry install --no-dev
```

5. Run train with the following command:

```
poetry run train -d <path to csv with data>
```

6.Run MLflow UI to see the information about experiments you conducted:

```
poetry run mlflow ui
```











