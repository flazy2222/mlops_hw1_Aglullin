# MLOps HW1 — Reproducible ML Pipeline (DVC + MLflow)

## Цель проекта
Создать воспроизводимый ML-пайплайн, включающий:
- версионирование данных через DVC;
- автоматическое выполнение этапов подготовки данных, обучения и оценки модели;
- логирование параметров, метрик и артефактов модели с помощью MLflow;
- воспроизводимость экспериментов одной командой.

---

## Как запустить проект (4 команды)

1. Клонировать репозиторий:
   ```bash
   git clone (https://github.com/flazy2222/mlops_hw1_Aglullin.git)
   cd mlops_hw1_Aglullin
2. Установить завимимости: 
pip install -r requirements.txt
3. Запустить весь ML-пайплайн:
dvc repro
4. Запустить MLflow UI:
mlflow ui --backend-store-uri sqlite:///mlflow.db


Краткое описание DVC-пайплайна
1) prepare

вход: data/raw/iris.csv

выполняет подготовку данных

выход: data/processed/iris_processed.csv

2) train

вход: обработанные данные + params.yaml

обучает модель LogisticRegression

логирует параметры, accuracy и модель в MLflow

выход: models/model.pkl

3) evaluate

вход: обученная модель + обработанные данные

вычисляет accuracy

выход: metrics.json


Где смотреть MLflow UI:
mlflow ui --backend-store-uri sqlite:///mlflow.db
http://127.0.0.1:5000
Эксперимент: iris_classification

Там можно увидеть:

параметры модели;

accuracy;

артефакты (model.pkl);

историю запусков.
