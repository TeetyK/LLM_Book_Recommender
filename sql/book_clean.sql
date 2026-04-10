CREATE TABLE books_cleaned (
    id SERIAL PRIMARY KEY,
    isbn13 TEXT REFERENCES books_raw(isbn13), -- เชื่อมกลับไปที่ตารางดิบ
    title TEXT,
    authors TEXT,
    categories TEXT,
    description TEXT,
    text_for_embedding TEXT, -- ข้อมูลที่รวมกันเพื่อเตรียมทำ Vector
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);