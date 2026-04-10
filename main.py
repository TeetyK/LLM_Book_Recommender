import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.messages import HumanMessage

from graph import app as graph_app

# --- Initial Setup ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# --- Streamlit UI ---
st.set_page_config(page_title="📚 AI Book Recommender", layout="wide")
st.title("📚 AI Book Recommender (RAG + LangGraph + Supabase)")

# For demo purposes, assigning a mock user_id to simulate user history
USER_ID = "mock_user_123"

if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติแชท
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ส่วนรับ Input
if user_input := st.chat_input("อยากอ่านแนวไหน บอกมาได้เลย หรือจะถามข้อมูล/สรุปข้อมูลก็ได้..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("กำลังประมวลผล..."):
            try:
                # Invoke the LangGraph application
                initial_state = {
                    "messages": [HumanMessage(content=user_input)],
                    "user_id": USER_ID
                }

                result = graph_app.invoke(initial_state)

                # Fetch the response generated from the graph's state
                response = result.get("response", "ขออภัย ไม่สามารถประมวลผลคำตอบได้")

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = f"เกิดข้อผิดพลาด: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
