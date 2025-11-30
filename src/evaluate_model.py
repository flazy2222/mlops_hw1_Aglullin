import sys
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import json


def main(input_path: str, model_path: str, metrics_path: str) -> None:
    df = pd.read_csv(input_path, header=None)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = joblib.load(model_path)
    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    metrics = {"accuracy": acc}
    print(f"Evaluate accuracy: {acc}")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    input_path = sys.argv[1]
    model_path = sys.argv[2]
    metrics_path = sys.argv[3]
    main(input_path, model_path, metrics_path)
