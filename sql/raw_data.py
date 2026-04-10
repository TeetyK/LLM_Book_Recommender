import os
import json
from dotenv import load_dotenv
from langchain_core.tools import tool
import psycopg2
from psycopg2 import extras
import pandas as pd
import numpy as np
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
connection = psycopg2.connect(DATABASE_URL)

if not DATABASE_URL:
    print("Warning: DATABASE_URL not set in .env")

raw_data = pd.read_csv(".\\datasets\\books.csv")
cols = ['isbn13','isbn10','title','subtitle','authors','categories','thumbnail','description','published_year','average_rating','num_pages','ratings_count']
df_subset = raw_data[cols].copy()
df_subset = df_subset.where(pd.notnull(df_subset), None)

data_values = [tuple(x) for x in df_subset.to_numpy()]

cur = connection.cursor()
insert_raw_data = f"INSERT INTO books_raw ({', '.join(cols)}) VALUES %s"
extras.execute_values(cur, insert_raw_data, data_values)

connection.commit()
cur.close()
connection.close()