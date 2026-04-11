import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_core.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Warning: SUPABASE_URL or SUPABASE_KEY not set in .env")

# Initialize Supabase client
def get_supabase_client() -> Client | None:
    if SUPABASE_URL and SUPABASE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    return None

def get_user_preferences(user_id: str) -> str:
    """
    Retrieves top interactions/preferences for a user from Supabase.
    Since we don't have the exact user history table, this currently returns a mock representation,
    but demonstrates how to query using the Supabase client.
    """
    client = get_supabase_client()
    if client:
        try:
            # Uncomment and adjust the following lines if a user transactions table exists:
            # response = client.table("user_transactions").select("book_category, book_title").eq("user_id", user_id).limit(5).execute()
            # if response.data:
            #     return str(response.data)
            pass
        except Exception as e:
            return f"Error fetching preferences: {e}"

    # Mocking preferences for the user as fallback
    return "User loves reading about Science Fiction, Technology, and History. Top books viewed: 'Dune', 'Sapiens'."

@tool
def query_database_tool(query_intent: str) -> str:
    """
    MCP Tool to query or summarize data from the database.
    Input should be a description of the data needed (e.g., 'summarize user interactions', 'list top 5 books').
    """
    if not DATABASE_URL:
        return "Database URL is not configured."

    try:
        db = SQLDatabase.from_uri(DATABASE_URL)
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )
        agent_executor = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True)
        response = agent_executor.invoke({"input": query_intent})
        return response.get("output", "Could not generate response.")
    except Exception as e:
        return f"Database Query Error: {e}"
