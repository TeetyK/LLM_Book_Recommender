import os
import json
import psycopg2
from psycopg2 import extras
import pandas as pd
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from nlp_utils import preprocess_text
from database import query_database_tool, get_user_preferences

DATABASE_URL = os.getenv("DATABASE_URL")

# --- 1. RAG Supabase Vector Retrieval ---

def get_similar_books_from_supabase(query_text: str, k: int = 5) -> str:
    """
    Embeds the user query using HuggingFace and retrieves the top-k similar
    books directly from Supabase's book_vectors table using pgvector's <=> operator.
    Applies dimension padding (to 3072) to match the database vectors.
    """
    if not DATABASE_URL:
        return "Error: DATABASE_URL not set."

    try:
        # 1. Initialize Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="mixedbread-ai/mxbai-embed-large-v1"
        )

        # 2. Embed the query
        query_vector = embeddings.embed_query(query_text)

        # 3. Apply padding to match 3072 dimensions
        target_dim = 3072
        if len(query_vector) < target_dim:
            query_vector = query_vector + [0.0] * (target_dim - len(query_vector))
        elif len(query_vector) > target_dim:
            query_vector = query_vector[:target_dim]

        # 4. Format for pgvector
        vector_str = "[" + ",".join([str(x) for x in query_vector]) + "]"

        # 5. Connect and Query
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        sql = f"""
            SELECT content, metadata
            FROM book_vectors
            ORDER BY embedding <=> %s
            LIMIT %s;
        """
        cur.execute(sql, (vector_str, k))
        rows = cur.fetchall()

        cur.close()
        conn.close()

        if not rows:
            return "No matching books found in the database."

        # 6. Format results
        context_parts = []
        for i, row in enumerate(rows):
            content = row[0]
            metadata = row[1] if row[1] else {}
            title = metadata.get("title", "Unknown Title") if isinstance(metadata, dict) else "Unknown"
            context_parts.append(f"Result {i+1}:\n{content}")

        return "\n\n".join(context_parts)
    except Exception as e:
        return f"Error querying vector database: {e}"

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
    Decides if the query is for data/summarization (SQL Tool) or Book Recommendation
    using an LLM intent classifier.
    """
    messages = state["messages"]
    latest_message = messages[-1].content

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    prompt = f"""You are an intent classifier for a book recommendation system.
Determine if the user's message is asking to query data (like database stats, summaries, or user info) or if they are asking for book recommendations.
Reply with EXACTLY one of the following words: "query_data" or "recommend_book".
Do not include any other text.

User's message: {latest_message}
Intent:"""

    response = llm.invoke(prompt)
    intent_str = response.content.strip().lower()

    if "query_data" in intent_str:
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

    # 2. NLP Preprocessing to reduce tokens (simplified)
    processed_query = preprocess_text(latest_message)

    # 3. RAG Retrieval from Supabase
    doc_context = get_similar_books_from_supabase(processed_query, k=5)

    if "Error" in doc_context:
        return {"response": doc_context}

    # 4. LLM Generation
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

    prompt = f"""You are an expert book recommender.
User Preferences/History Context: {user_context}
Database Results Context:
{doc_context}
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
