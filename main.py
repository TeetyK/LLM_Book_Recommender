import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from graph import app as graph_app
from database import login_or_register

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key: genai.configure(api_key=api_key)

st.set_page_config(page_title="📚 AI Book Recommender", layout="wide")

# ==========================================
# 1. ระบบ Login (Sidebar)
# ==========================================
with st.sidebar:
    st.header("🔐 เข้าสู่ระบบ")
    input_username = st.text_input("ชื่อผู้ใช้ (Username):", placeholder="เช่น teety_01")
    login_btn = st.button("เข้าสู่ระบบ")

    if login_btn and input_username:
        # ไปเช็คใน DB ว่ามีชื่อนี้ไหม ไม่มีก็สร้างให้เลย
        user_id = login_or_register(input_username)
        if user_id:
            st.session_state["logged_in"] = True
            st.session_state["username"] = input_username
            st.session_state["user_id"] = user_id
            st.success(f"ยินดีต้อนรับ {input_username}!")
        else:
            st.error("เกิดข้อผิดพลาดในการเชื่อมต่อฐานข้อมูล")

# เช็คสิทธิ์ก่อนแสดงช่องแชท
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.title("📚 AI Book Recommender")
    st.warning("👈 โปรดพิมพ์ชื่อผู้ใช้ที่เมนูด้านซ้ายเพื่อเริ่มใช้งานครับ (พิมพ์ชื่ออะไรก็ได้ ระบบจะจำไว้ให้ครับ)")
    st.stop()

# ==========================================
# 2. หน้าจอ Chat
# ==========================================
USER_ID = st.session_state["user_id"]
USERNAME = st.session_state["username"]

st.title(f"📚 AI Book Recommender (👤 {USERNAME})")

if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_input := st.chat_input("อยากอ่านแนวไหน หรืออยากค้นหาข้อมูลสถิติ ถามมาได้เลย..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาจากฐานข้อมูล..."):
            try:
                # ส่ง ID ของผู้ใช้เข้า AI
                initial_state = {"messages": [HumanMessage(content=user_input)], "user_id": USER_ID}
                result = graph_app.invoke(initial_state)
                response = result.get("response", "ขออภัย ประมวลผลล้มเหลว")

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")