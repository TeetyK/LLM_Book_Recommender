import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from nlp_utils import preprocess_text
from database import get_similar_books, get_user_preferences, manage_postgres_data , add_prompt_history

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "messages"]
    user_id: int
    intent: str
    response: str

# ==========================================
# ระบบ Fallback (Gemini -> Ollama)
# ==========================================
def call_llm_with_fallback(prompt_or_messages, tools=None):
    """
    ลองใช้ Gemini ก่อน ถ้า Error หรือ Quota เต็ม ให้สลับไปใช้ Ollama อัตโนมัติ
    """
    try:
        # ลองเรียก Gemini
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )
        if tools:
            llm = llm.bind_tools(tools)
            
        return llm.invoke(prompt_or_messages)
        
    except Exception as e:
        print(f"\n⚠️ Gemini ไม่พร้อมใช้งาน ({e})\n🔄 กำลังสลับไปใช้ Local Ollama (book_qwen)...")
        
        llm_ollama = ChatOllama(
            model="book_qwen",
            temperature=0.3
        )
        if tools:
            llm_ollama = llm_ollama.bind_tools(tools)
            
        return llm_ollama.invoke(prompt_or_messages)

# ==========================================
# Nodes ของ LangGraph
# ==========================================
def router_node(state: GraphState):
    """วิเคราะห์ความตั้งใจ และบันทึกประวัติทันทีที่ข้อความเข้าระบบ"""
    messages = state["messages"]
    latest_message = messages[-1].content
    user_id = state.get("user_id")

    # 🌟 จุดแก้สำคัญ: บันทึกประวัติตั้งแต่จุดเริ่มต้นของ Graph
    if user_id:
        print(f"DEBUG: Saving prompt for User {user_id}: {latest_message}")
        add_prompt_history(user_id, latest_message)

    # ส่วนของ Logic การแยก Intent เดิม
    msg_lower = latest_message.lower()
    db_keywords = ["summarize", "data", "database", "ค้นหาข้อมูล", "เรทติ้ง", "สรุป"]
    if any(keyword in msg_lower for keyword in db_keywords):
        return {"intent": "query_data"}
    return {"intent": "recommend_book"}

def route_intent(state: GraphState):
    if state["intent"] == "query_data": return "sql_query_node"
    return "recommendation_node"

def sql_query_node(state: GraphState):
    latest_message = state["messages"][-1].content
    
    prompt = f"User request: '{latest_message}'. Use manage_postgres_data tool to find the answer."
    
    # ใช้ฟังก์ชัน Fallback ของเราแทนการเรียก LLM ตรงๆ (พร้อมแนบ Tools)
    ai_msg = call_llm_with_fallback(prompt, tools=[manage_postgres_data])
    
    if ai_msg.tool_calls:
        tool_call = ai_msg.tool_calls[0]
        tool_result = manage_postgres_data.invoke(tool_call["args"])
        
        summary_prompt = f"คำถามผู้ใช้: {latest_message}\nผลจาก Database SQL: {tool_result}\nจงสรุปคำตอบให้ผู้ใช้อ่านเข้าใจง่าย"
        # ใช้ Fallback ตอนสรุปผลด้วย (เผื่อพังตอนจะสรุป)
        final_answer = call_llm_with_fallback(summary_prompt)
        return {"response": final_answer.content}
    
    return {"response": ai_msg.content}

def recommendation_node(state: GraphState):
    latest_message = state["messages"][-1].content
    user_id = state.get("user_id")

    # 🌟 1. แอบบันทึก Prompt ของผู้ใช้ลงฐานข้อมูล (ทำก่อนเลย!)
    if user_id:
        add_prompt_history(user_id, latest_message)

    # 2. ดึงประวัติของคนนี้มาดู (ตอนนี้มันจะรวมถึงสิ่งที่เพิ่งพิมพ์เมื่อกี้ด้วย!)
    user_context = get_user_preferences(user_id) if user_id else "ไม่มีข้อมูลประวัติ"

    # 3. เตรียม NLP และค้นหา Vector
    processed_query = preprocess_text(latest_message)
    search_query = processed_query if processed_query else latest_message 

    # (ตรงนี้คือโค้ดเดิมที่เรียกใช้ get_similar_books และ AI ตามปกติ)
    doc_context = get_similar_books(search_query, user_id=user_id, k=5)

    prompt = f"""ความสนใจของผู้ใช้: {user_context}
ข้อมูลหนังสือจาก Database: {doc_context}
คำถามของผู้ใช้: {latest_message}

หน้าที่ของคุณคือเลือกหนังสือจาก 'ข้อมูลหนังสือ' ให้ตอบโจทย์คำถามผู้ใช้โดยอิงจาก 'ความสนใจของผู้ใช้' ด้วย (ถ้ามี)"""

    response = call_llm_with_fallback(prompt)
    return {"response": response.content}

# ==========================================
# สร้าง Workflow
# ==========================================
workflow = StateGraph(GraphState)
workflow.add_node("router", router_node)
workflow.add_node("sql_query_node", sql_query_node)
workflow.add_node("recommendation_node", recommendation_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route_intent)
workflow.add_edge("sql_query_node", END)
workflow.add_edge("recommendation_node", END)

app = workflow.compile()