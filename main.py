import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
import time
os.environ["LANGCHAIN_GOOGLE_FIREBASE_API_VERSION"] = "v1" 
os.environ["GOOGLE_API_VERSION"] = "v1"
# --- Initial Setup ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
RAW_DATA_PATH = 'datasets/books.csv'
CLEANED_DATA_PATH = 'datasets/books_cleaned.csv'
VECTORSTORE_PATH = 'chroma_db_google'

# --- 1. Data Preprocessing ---
def preprocess_books_data():
    if not os.path.exists(RAW_DATA_PATH):
        st.error(f"ไม่พบไฟล์ {RAW_DATA_PATH}")
        return False
    
    st.write("กำลังเตรียมข้อมูลหนังสือ...")
    df = pd.read_csv(RAW_DATA_PATH, on_bad_lines='warn').head(5) # ทดสอบ 500 เล่มก่อนถ้าเครื่องช้า
    
    # รวมข้อความสำหรับทำ Embedding
    df['text_for_embedding'] = df.apply(lambda r: 
        f"Title: {r.get('title','')}\nAuthors: {r.get('authors','')}\n"
        f"Categories: {r.get('categories','')}\nDesc: {r.get('description','')}", axis=1)
    
    os.makedirs('datasets', exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    return True

# --- 2. Vector Store Management ---
@st.cache_resource
def get_retriever():
    if not os.path.exists(CLEANED_DATA_PATH):
        if not preprocess_books_data(): return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    if not os.path.exists(VECTORSTORE_PATH):
        st.warning("กำลังสร้างฐานข้อมูล... (กำลังแบ่งส่งข้อมูลเพื่อไม่ให้เกิน Quota)")
        df = pd.read_csv(CLEANED_DATA_PATH)
        texts = df['text_for_embedding'].fillna("Unknown").tolist()
        metadatas = df[['title', 'authors']].to_dict('records')

        vectorstore = Chroma(
            persist_directory=VECTORSTORE_PATH, 
            embedding_function=embeddings
        )
        batch_size = 20 
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            
            vectorstore.add_texts(texts=batch_texts, metadatas=batch_meta)
            
            st.write(f"บันทึกแล้ว {min(i + batch_size, len(texts))} / {len(texts)} เล่ม...")
            time.sleep(5) 

        st.success("สร้างฐานข้อมูลเสร็จสมบูรณ์!")
    else:
        vectorstore = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# --- 3. RAG Chain Construction (New Style) ---
def get_recommender_chain():
    retriever = get_retriever()
    if not retriever: return None

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash", 
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

    template = """You are an expert book recommender. Use the provided context to suggest books.
    Context: {context}
    Question: {question}
    
    Answer in a friendly tone with bullet points:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # สร้าง Chain ด้วย LCEL (แทนที่ RetrievalQA)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- 4. Streamlit UI ---
st.set_page_config(page_title="📚 AI Book Recommender", layout="wide")
st.title("📚 AI Book Recommender (RAG)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติแชท
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ส่วนรับ Input
if user_input := st.chat_input("อยากอ่านแนวไหน บอกมาได้เลย..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    chain = get_recommender_chain()
    if chain:
        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาหนังสือที่เหมาะกับคุณ..."):
                response = chain.invoke(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})