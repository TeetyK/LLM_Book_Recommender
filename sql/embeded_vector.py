import os
import time
import json
import pandas as pd
import psycopg2
from psycopg2 import extras
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",google_api_key=api_key)
embeddings = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1"
)
DATABASE_URL = os.getenv("DATABASE_URL")
conn = psycopg2.connect(DATABASE_URL)

cur = conn.cursor()

print("กำลังสร้าง Vector Database ใน Supabase...")
cur.execute("SELECT isbn13 FROM book_vectors")
existing_isbns = set(row[0] for row in cur.fetchall())
print(f"✅ พบข้อมูลที่ทำไปแล้ว: {len(existing_isbns)} เล่ม")

df_all = pd.read_sql("SELECT isbn13, text_for_embedding, title, authors FROM books_cleaned", conn)
df_todo = df_all[~df_all['isbn13'].isin(existing_isbns)].copy()

if df_todo.empty:
    print("✨ ข้อมูลทั้งหมดถูกทำเป็น Vector เรียบร้อยแล้ว! ไม่ต้องทำเพิ่ม")
else:
    print(f"🚀 เหลือที่ต้องทำเพิ่มอีก: {len(df_todo)} เล่ม")
    
    batch_size = 50 # HuggingFace รันในเครื่อง ปรับให้ใหญ่ขึ้นได้
    target_dim = 3072 # มิติที่คุณต้องการใน SQL

    for i in range(0, len(df_todo), batch_size):
        batch_df = df_todo.iloc[i : i + batch_size]
        texts = batch_df['text_for_embedding'].fillna("Unknown").tolist()
        
        # คำนวณ Embedding
        vector_embeddings = embeddings.embed_documents(texts)
        
        # ระบบจัดการ Dimension ให้ได้ 3072 (Padding)
        final_vectors = []
        for v in vector_embeddings:
            if len(v) < target_dim:
                v = v + [0.0] * (target_dim - len(v))
            elif len(v) > target_dim:
                v = v[:target_dim]
            final_vectors.append(v)

        # เตรียมข้อมูล Insert
        batch_data = []
        for j, row in enumerate(batch_df.to_dict('records')):
            meta = json.dumps({"title": row['title'], "authors": row['authors']})
            batch_data.append((
                row['isbn13'],
                row['text_for_embedding'],
                final_vectors[j],
                meta
            ))
        
        # ยิงเข้า SQL
        insert_query = "INSERT INTO book_vectors (isbn13, content, embedding, metadata) VALUES %s"
        extras.execute_values(cur, insert_query, batch_data)
        conn.commit()
        
        print(f"📊 Progress: ทำสำเร็จเพิ่ม {i + len(batch_df)} / {len(df_todo)} เล่ม")

print("🎉 กระบวนการเสร็จสมบูรณ์!")
cur.close()
conn.close()