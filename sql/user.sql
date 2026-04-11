CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,              -- ไอดีผู้ใช้ (รันอัตโนมัติ)
    username VARCHAR(50) UNIQUE NOT NULL,    -- ชื่อผู้ใช้สำหรับ Login (ห้ามซ้ำ)
    email VARCHAR(100) UNIQUE,               -- อีเมล (เผื่อทำระบบลืมรหัสผ่าน)
    password_hash TEXT,                      -- รหัสผ่าน (ควรเข้ารหัสก่อนเก็บเสมอ)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE user_history (
    history_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE, -- เชื่อมกับตาราง users
    isbn13 TEXT,                             -- รหัสหนังสือ (เชื่อมกับตารางหนังสือที่คุณมี)
    book_title TEXT NOT NULL,                -- ชื่อหนังสือ (เก็บไว้เพื่อง่ายต่อการให้ AI ดึงไปอ่าน)
    book_category TEXT,                      -- หมวดหมู่ (เช่น 'Science Fiction', 'Business')
    interaction_type VARCHAR(20) DEFAULT 'viewed', -- ประเภท: 'viewed' (เข้าดู), 'read' (อ่านจบ), 'liked' (กดใจ)
    rating INTEGER CHECK (rating >= 1 AND rating <= 5), -- (Optional) คะแนนรีวิว 1-5 ดาว
    interacted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_history_user_id ON user_history(user_id);