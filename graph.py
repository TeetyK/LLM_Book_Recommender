import os
import pandas as pd
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from nlp_utils import preprocess_text
from database import query_database_tool, get_user_preferences

# Constants from original main.py
RAW_DATA_PATH = 'datasets/books.csv'
CLEANED_DATA_PATH = 'datasets/books_cleaned.csv'
VECTORSTORE_PATH = 'chroma_db_google'

# --- 1. RAG Setup Logic Moved from main.py ---
def preprocess_books_data():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Not found {RAW_DATA_PATH}")
        return False

    print("กำลังเตรียมข้อมูลหนังสือ...")
    df = pd.read_csv(RAW_DATA_PATH, on_bad_lines='warn').head(5)
    df['text_for_embedding'] = df.apply(lambda r:
        f"Title: {r.get('title','')}\nAuthors: {r.get('authors','')}\n"
        f"Categories: {r.get('categories','')}\nDesc: {r.get('description','')}", axis=1)

    os.makedirs('datasets', exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    return True

def get_retriever():
    if not os.path.exists(CLEANED_DATA_PATH):
        if not preprocess_books_data(): return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    if not os.path.exists(VECTORSTORE_PATH):
        print("กำลังสร้างฐานข้อมูล... (กำลังแบ่งส่งข้อมูลเพื่อไม่ให้เกิน Quota)")
        df = pd.read_csv(CLEANED_DATA_PATH)
        texts = df['text_for_embedding'].fillna("Unknown").tolist()
        metadatas = df[['title', 'authors']].to_dict('records')

        vectorstore = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=embeddings
        )
        batch_size = 20
        import time
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            vectorstore.add_texts(texts=batch_texts, metadatas=batch_meta)
            time.sleep(5)

        print("สร้างฐานข้อมูลเสร็จสมบูรณ์!")
    else:
        vectorstore = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 5})

# --- 2. LangGraph Setup ---
# Define State Schema
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "messages"]
    user_id: str
    intent: str
    response: str

# Define Node: Intent Classification
def router_node(state: GraphState):
    """
    Decides if the query is for data/summarization (SQL Tool) or Book Recommendation.
    """
    messages = state["messages"]
    latest_message = messages[-1].content.lower()

    # Simple keyword-based routing. Can be upgraded to LLM-based routing.
    if any(keyword in latest_message for keyword in ["summarize", "data", "transactions", "database", "สรุปข้อมูล", "ข้อมูล user"]):
        intent = "query_data"
    else:
        intent = "recommend_book"

    return {"intent": intent}

def route_intent(state: GraphState):
    if state["intent"] == "query_data":
        return "sql_query_node"
    return "recommendation_node"

# Define Node: SQL Query Tool
def sql_query_node(state: GraphState):
    messages = state["messages"]
    latest_message = messages[-1].content

    # Call the MCP Tool
    result = query_database_tool.invoke({"query_intent": latest_message})

    return {"response": result}

# Define Node: Recommendation (RAG + User Context)
def recommendation_node(state: GraphState):
    messages = state["messages"]
    latest_message = messages[-1].content
    user_id = state.get("user_id", "default_user")

    # 1. Fetch User Preferences (Context)
    user_context = get_user_preferences(user_id)

    # 2. NLP Preprocessing to reduce tokens
    processed_query = preprocess_text(latest_message)

    # 3. RAG Retrieval
    retriever = get_retriever()
    if not retriever:
        return {"response": "Error: Database not ready."}

    docs = retriever.invoke(processed_query)
    doc_context = "\n\n".join([doc.page_content for doc in docs])

    # 4. LLM Generation
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

    prompt = f"""You are an expert book recommender.
User Preferences/History Context: {user_context}
Database Results Context: {doc_context}
User Question: {latest_message}

Using both the user's history and the database context, suggest books. Answer in a friendly tone with bullet points:"""

    response = llm.invoke(prompt)
    return {"response": response.content}


# --- Build Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("router", router_node)
workflow.add_node("sql_query_node", sql_query_node)
workflow.add_node("recommendation_node", recommendation_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    route_intent,
    {
        "sql_query_node": "sql_query_node",
        "recommendation_node": "recommendation_node",
    }
)

workflow.add_edge("sql_query_node", END)
workflow.add_edge("recommendation_node", END)

app = workflow.compile()
