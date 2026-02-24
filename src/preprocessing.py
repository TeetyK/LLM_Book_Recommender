import pandas as pd

def preprocessing(path):
    df = pd.read_csv(path)
    print(df)
    return df
