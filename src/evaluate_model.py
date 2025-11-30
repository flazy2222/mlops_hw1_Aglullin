import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import json
import sys

# Аргументы: путь к данным и к модели
input_path = sys.argv[1]
model_path = sys.argv[2]
metrics_path = sys.argv[3]

# Загружаем данные и модель
data = pd.read_csv(input_path)
model = joblib.load(model_path)

X = data.drop("species", axis=1)
y = data["species"]

# Предсказания
y_pred = model.predict(X)

# Вычисляем метрики
metrics = {"accuracy": accuracy_score(y, y_pred)}

# Сохраняем в JSON
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Metrics saved to {metrics_path}")
