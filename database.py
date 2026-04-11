import os
import json
import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """เชื่อมต่อ PostgreSQL ผ่าน psycopg2"""
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# ==========================================
# ส่วนจัดการ User และประวัติการอ่าน
# ==========================================
def login_or_register(username: str):
    """ตรวจสอบผู้ใช้ ถ้าไม่มีให้สร้างใหม่ (Auto-Register)"""
    conn = get_db_connection()
    if not conn: return None
    try:
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        # เช็คว่ามี username นี้หรือยัง
        cur.execute("SELECT user_id FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        
        if user:
            user_id = user['user_id']
        else:
            # ถ้าไม่มี สร้างใหม่เลย
            cur.execute("INSERT INTO users (username) VALUES (%s) RETURNING user_id", (username,))
            user_id = cur.fetchone()['user_id']
            conn.commit()
            
        cur.close()
        conn.close()
        return user_id
    except Exception as e:
        print(f"Login Error: {e}")
        return None

def get_user_preferences(user_id: int) -> str:
    """ดึงประวัติความสนใจของผู้ใช้จากตาราง user_history จริง"""
    conn = get_db_connection()
    if not conn: return "ไม่มีข้อมูลความสนใจ (ฐานข้อมูลมีปัญหา)"

    try:
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        sql = """
            SELECT book_category, book_title 
            FROM user_history 
            WHERE user_id = %s 
            ORDER BY interacted_at DESC LIMIT 5
        """
        cur.execute(sql, (user_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return "ผู้ใช้นี้เป็นผู้ใช้ใหม่ ยังไม่มีประวัติการอ่าน"

        history = [f"เคยอ่าน '{r['book_title']}' (หมวด {r['book_category']})" for r in rows]
        return "ประวัติความสนใจ: " + ", ".join(history)

    except Exception as e:
        print(f"Error fetching preferences: {e}")
        return "ไม่มีข้อมูลความสนใจ (พบข้อผิดพลาด)"

@tool
def manage_postgres_data(action: str, table_name: str, match_column: str = None, match_value: str = None) -> str:
    """
    เครื่องมือค้นหาข้อมูลสถิติหรือรายชื่อหนังสือจาก Database ตรงๆ
    Args:
        action (str): คำสั่ง 'select'
        table_name (str): ชื่อตาราง ('books_raw' หรือ 'books_cleaned')
        match_column (str, optional): คอลัมน์ที่ต้องการค้นหา (เช่น 'title')
        match_value (str, optional): ค่าที่จะค้นหา
    """
    conn = get_db_connection()
    if not conn: return "Error: Database not connected."

    try:
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        cols = "*"
        if table_name == "books_raw": cols = "title, authors, average_rating, published_year"
        if table_name == "books_cleaned": cols = "title, authors, categories, description"
        
        if action == "select":
            if match_column and match_value:
                sql = f"SELECT {cols} FROM {table_name} WHERE {match_column} ILIKE %s LIMIT 5"
                cur.execute(sql, (f"%{match_value}%",))
            else:
                sql = f"SELECT {cols} FROM {table_name} LIMIT 5"
                cur.execute(sql)
                
            rows = cur.fetchall()
            cur.close()
            conn.close()
            
            if not rows:
                return f"ไม่พบข้อมูลใน {table_name} สำหรับ {match_column}={match_value}"
            return json.dumps(rows, ensure_ascii=False)
        return "Error: Unsupported action."
    except Exception as e:
        return f"Database Error: {str(e)}"
def add_user_history(user_id: int, book_title: str, category: str, interaction_type: str = 'recommended'):
    """ฟังก์ชันสำหรับบันทึกประวัติความสนใจลงฐานข้อมูล"""
    conn = get_db_connection()
    if not conn: return
    try:
        cur = conn.cursor()
        sql = """
            INSERT INTO user_history (user_id, book_title, book_category, interaction_type) 
            VALUES (%s, %s, %s, %s)
        """
        cur.execute(sql, (user_id, book_title, category, interaction_type))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error inserting history: {e}")

# ========================================================
# แก้ไขฟังก์ชัน get_similar_books เดิม ให้รับ user_id ด้วย
# ========================================================
def get_similar_books(query_text: str, user_id: int = None, k: int = 5) -> str:
    """RAG: ค้นหา Vector พร้อมเติม Padding ให้ครบ 3072 มิติ"""
    conn = get_db_connection()
    if not conn: return "Error: Database connection failed."

    try:
        embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
        query_vector = embeddings.embed_query(query_text)
        
        # 🌟 เพิ่มส่วนนี้: เติม 0 (Padding) ให้ครบ 3072 มิติ เพื่อให้ตรงกับใน Database
        target_dim = 3072
        if len(query_vector) < target_dim:
            query_vector = query_vector + [0.0] * (target_dim - len(query_vector))
        elif len(query_vector) > target_dim:
            query_vector = query_vector[:target_dim]

        # แปลงเป็น String สำหรับ pgvector
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"

        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        sql = """
            SELECT content, metadata 
            FROM book_vectors 
            ORDER BY embedding <=> %s::vector 
            LIMIT %s
        """
        cur.execute(sql, (vector_str, k))
        rows = cur.fetchall()

        result_texts = []
        for index, row in enumerate(rows):
            meta = row['metadata'] if row['metadata'] else {}
            title = meta.get('title', 'Unknown')
            authors = meta.get('authors', 'Unknown')
            category = meta.get('categories', 'General')
            
            result_texts.append(f"Title: {title} | Authors: {authors} | Content: {row['content']}")
            
            # บันทึกประวัติ
            if user_id and index == 0 and title != 'Unknown':
                add_user_history(user_id, title, category, 'recommended')
            
        cur.close()
        conn.close()
        return "\n\n".join(result_texts)
    except Exception as e:
        return f"Vector search error: {e}"
    
def add_prompt_history(user_id: int, prompt_text: str):
    """ฟังก์ชันสำหรับบันทึกประวัติ 'คำถาม/คำค้นหา' ของผู้ใช้"""
    conn = get_db_connection()
    if not conn: return
    try:
        cur = conn.cursor()
        
        # ตัดข้อความให้เหลือแค่ 100 ตัวอักษร ป้องกันคนพิมพ์ยาวเกินไปจน Database รก
        short_prompt = prompt_text[:100]
        
        # บันทึกลงตาราง โดยใช้ book_title เก็บคำค้นหา และตั้งหมวดหมู่เป็น 'User Prompt'
        sql = """
            INSERT INTO user_history (user_id, book_title, book_category, interaction_type) 
            VALUES (%s, %s, %s, %s)
        """
        # ใส่ Prefix คำว่า "เคยพิมพ์หา:" นำหน้า เพื่อให้ AI เข้าใจง่ายๆ ตอนดึงประวัติมาอ่าน
        cur.execute(sql, (user_id, f"เคยพิมพ์หา: {short_prompt}", "User Prompt", "searched"))
        
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error inserting prompt history: {e}")
    