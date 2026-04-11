import re
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
import nltk
from nltk.corpus import stopwords

# โหลด Stop words ภาษาอังกฤษเผื่อไว้ (สำหรับชื่อหนังสือหรือคำทับศัพท์)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def preprocess_text(text: str) -> str:
    """
    ทำความสะอาดและตัดคำภาษาไทย + อังกฤษ เพื่อลด Token สำหรับ RAG
    """
    if not text:
        return ""

    # 1. ลบอักขระพิเศษ แต่เก็บตัวอักษรไทย (ก-๙), อังกฤษ และตัวเลขไว้
    text = re.sub(r'[^\w\sก-๙]', '', text.lower())

    # 2. ตัดคำด้วย PyThaiNLP (engine='newmm' คือมาตรฐานที่ดีที่สุดตอนนี้)
    # keep_whitespace=False เพื่อตัดช่องว่างทิ้ง
    tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)

    # 3. เตรียมชุดคำที่ไม่มีประโยชน์ (Stop words) ทั้งไทยและอังกฤษ
    thai_stops = set(thai_stopwords())
    eng_stops = set(stopwords.words('english'))
    all_stops = thai_stops.union(eng_stops)

    # 4. กรองคำที่อยู่ใน Stop words ออก
    clean_tokens = [word for word in tokens if word not in all_stops]

    return " ".join(clean_tokens)