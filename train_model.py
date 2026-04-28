import pandas as pd
import numpy as np
import joblib
import mlflow

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature


# загрузка данных
df = pd.read_csv("df_clear.csv")

# признаки и target
X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

# масштабирование
scaler = StandardScaler()
X = scaler.fit_transform(X)

pt = PowerTransformer()
y = pt.fit_transform(y.values.reshape(-1, 1))

# разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# параметры модели
params = {
    "alpha": [0.0001, 0.001],
    "penalty": ["l2", "elasticnet"],
    "loss": ["squared_error", "huber"]
}

# MLflow
mlflow.set_experiment("housing_model")

with mlflow.start_run():

    model = SGDRegressor(random_state=42)
    grid = GridSearchCV(model, params, cv=3)
    grid.fit(X_train, y_train.ravel())

    best = grid.best_estimator_

    # предсказание
    pred = best.predict(X_test)

    # обратное преобразование
    pred = pt.inverse_transform(pred.reshape(-1, 1))
    real = pt.inverse_transform(y_test)

    # метрики
    rmse = np.sqrt(mean_squared_error(real, pred))
    mae = mean_absolute_error(real, pred)
    r2 = r2_score(real, pred)

    # логирование
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # сохранение модели
    mlflow.sklearn.log_model(
        best,
        "model",
        signature=infer_signature(X_train, best.predict(X_train))
    )

    joblib.dump(best, "model.pkl")

    # файл с моделью (для Jenkins)
    with open("best_model.txt", "w") as f:
        f.write("model.pkl")

print("Model trained and saved")