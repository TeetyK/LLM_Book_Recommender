import os
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
conn = psycopg2.connect(DATABASE_URL)

query_select = "SELECT * FROM books_raw"
df = pd.read_sql(query_select, conn)

df['published_year'] = pd.to_numeric(df['published_year'], errors='coerce')
df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce')
df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')

df["missing_description"] = np.where(df["description"].isna(), 1, 0)
df["age_of_book"] = 2024 - df["published_year"]
df = df.dropna(subset=["description", "num_pages", "average_rating", "published_year"])

df['words_in_description'] = df['description'].str.split().str.len()
df_final = df[df['words_in_description'] >= 25].copy()

df_final['title_and_subtitle'] = np.where(
    df_final["subtitle"].isna() | (df_final["subtitle"] == ""),
    df_final["title"],
    df_final["title"].astype(str) + ": " + df_final["subtitle"].astype(str)
)

df_final["tagged_description"] = df_final["isbn13"].astype(str) + " " + df_final["description"].astype(str)

cols_to_drop = ["subtitle","missing_description","age_of_book","words_in_description"]
df_final = df_final.drop(columns=cols_to_drop, errors='ignore')
df_to_db = df_final.where(pd.notnull(df_final), None)

df_final['text_for_embedding'] = df_final.apply(lambda r:
        f"Title: {r.get('title','')}\nAuthors: {r.get('authors','')}\n"
        f"Categories: {r.get('categories','')}\nDesc: {r.get('description','')}", axis=1)

cur = conn.cursor()
columns = [
    'isbn13', 
    'title', 
    'authors', 
    'categories', 
    'description', 
    'text_for_embedding'
]
df_final = df_final[columns].copy()
data_values = [tuple(x) for x in df_final.to_numpy()]
insert_query = f"INSERT INTO books_cleaned ({', '.join(columns)}) VALUES %s"

try:
    # cur.execute("TRUNCATE TABLE books_cleaned")     
    extras.execute_values(cur, insert_query, data_values)
    conn.commit()
    print(f"Cleaned and Inserted {len(data_values)} rows into books_cleaned!")
except Exception as e:
    print(f"Error: {e}")
    conn.rollback()
finally:
    cur.close()
    conn.close()