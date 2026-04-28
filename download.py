import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def prepare_data():
    df = pd.read_csv("housing.csv")

    # удаляем пустые значения
    df = df.dropna()

    # категориальные признаки
    cat = ["ocean_proximity"]

    enc = OrdinalEncoder()
    df[cat] = enc.fit_transform(df[cat])

    # сохраняем
    df.to_csv("df_clear.csv", index=False)

if __name__ == "__main__":
    prepare_data()