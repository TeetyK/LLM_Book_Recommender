 CREATE EXTENSION IF NOT EXISTS vector;

 CREATE TABLE book_vectors (
    id BIGSERIAL PRIMARY KEY,
    isbn13 TEXT REFERENCES books_raw(isbn13),
    content TEXT, -- เก็บ text_for_embedding
    embedding VECTOR(768), -- 768 มิติ สำหรับ model 'gemini-embedding-001'
    metadata JSONB -- เก็บ title, authors ไว้แสดงผลเร็วๆ
);