import sys
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


def main(input_path: str, model_path: str) -> None:
    # У Iris с UCI нет заголовков → читаем без header
    df = pd.read_csv(input_path, header=None)

    # Последний столбец — таргет, остальные — признаки
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"Train accuracy: {acc}")

    # Создаём папку models/, если её нет
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Сохраняем модель
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    input_path = sys.argv[1]
    model_path = sys.argv[2]
    main(input_path, model_path)
