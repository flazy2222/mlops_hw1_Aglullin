import sys
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import yaml


def main(input_path: str, model_path: str) -> None:
    # Загружаем гиперпараметры из params.yaml
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train_params = params.get("train", {})
    max_iter = train_params.get("max_iter", 200)
    random_state = train_params.get("random_state", 42)

    # Настраиваем MLflow (локальная БД в файле mlflow.db)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris_classification")

    # Загружаем данные
    df = pd.read_csv(input_path, header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Модель
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)

    # Запускаем MLflow run
    with mlflow.start_run():
        model.fit(X, y)

        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        print(f"Train accuracy: {acc}")

        # Логируем параметры
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", random_state)

        # Логируем путь к данным и модели (полезно для воспроизводимости)
        mlflow.log_param("input_path", input_path)
        mlflow.log_param("model_path", model_path)

        # Логируем метрику
        mlflow.log_metric("accuracy", acc)

        # Сохраняем модель на диск
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Логируем модель как артефакт
        mlflow.log_artifact(model_path)


if __name__ == "__main__":
    input_path = sys.argv[1]      # путь к data/processed/iris_processed.csv
    model_path = sys.argv[2]      # путь к models/model.pkl
    main(input_path, model_path)

