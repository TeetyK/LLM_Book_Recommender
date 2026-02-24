import pandas as pd

def preprocessing(path:str,file:str)->pd.DataFrame:
    df = pd.read_csv(path+file)
    print(df.head())
    df['tagged_description'].to_csv(path+"tagged_description.txt")
    return df
