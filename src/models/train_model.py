import sys
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow


def main(input_path: str, model_path: str) -> None:
    # Настраиваем MLflow (локальная БД в файле mlflow.db)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris_classification")

    df = pd.read_csv(input_path, header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = LogisticRegression(max_iter=200)

    # Запускаем MLflow run
    with mlflow.start_run():
        model.fit(X, y)

        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        print(f"Train accuracy: {acc}")

        # Логируем параметры
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", model.max_iter)

        # Логируем метрику
        mlflow.log_metric("accuracy", acc)

        # Сохраняем модель на диск
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Логируем модель как артефакт
        mlflow.log_artifact(model_path)
if __name__ == "__main__":
    input_path = sys.argv[1]
    model_path = sys.argv[2]
    main(input_path, model_path)
