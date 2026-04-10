CREATE TABLE books_raw (
    isbn13 TEXT PRIMARY KEY, -- ใช้เป็น PK เพราะเป็นมาตรฐานสากล
    isbn10 TEXT,
    title TEXT NOT NULL,
    subtitle TEXT,
    authors TEXT, -- เก็บเป็น TEXT หรือใช้ ARRAY ถ้าฐานข้อมูลรองรับ
    categories TEXT,
    thumbnail TEXT,
    description TEXT,
    published_year FLOAT,
    average_rating FLOAT,
    num_pages FLOAT,
    ratings_count FLOAT,
     created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);