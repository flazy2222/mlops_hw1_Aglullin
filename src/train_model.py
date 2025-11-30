import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

data = pd.read_csv(input_path)

X = data.drop("species", axis=1)  # Признаки
y = data["species"]               # Целевая переменная

# Обучаем модель
model = RandomForestClassifier()
model.fit(X, y)

# Сохраняем модель
joblib.dump(model, output_path)
print(f"Model saved to {output_path}")
