import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Warning: SUPABASE_URL or SUPABASE_KEY not set in .env")

# Initialize Supabase client
def get_supabase_client() -> Client | None:
    if SUPABASE_URL and SUPABASE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    return None

def get_user_preferences(user_id: str) -> str:
    """
    Mock function to retrieve top K interactions/preferences for a user from Supabase.
    Since we don't have the exact database schema, this returns a mock representation.
    In a real scenario, this would query the transactions/interactions table.
    """
    # client = get_supabase_client()
    # if client:
    #     try:
    #         response = client.table("user_transactions").select("book_category, book_title").eq("user_id", user_id).limit(5).execute()
    #         return str(response.data)
    #     except Exception as e:
    #         return f"Error fetching preferences: {e}"

    # Mocking preferences for the user
    return "User loves reading about Science Fiction, Technology, and History. Top books viewed: 'Dune', 'Sapiens'."

@tool
def query_database_tool(query_intent: str) -> str:
    """
    MCP Tool to query or summarize user data, books, and transactions using Supabase.
    Input should be a description of the data needed (e.g., 'summarize user interactions', 'list top 5 books').
    """
    client = get_supabase_client()
    if not client:
        return "Supabase connection is not configured."

    # Mocking the MCP logic based on query_intent to avoid direct SQL execution without knowing the exact schema
    # In a real implementation, you could use Langchain's SQLDatabase with SUPABASE_DB_URL
    if "summarize" in query_intent.lower() and "user" in query_intent.lower():
        return "Summary: We have 500 active users. Most users are interested in Technology and Fiction."
    elif "book" in query_intent.lower():
        return "Top books currently are: 'The Martian', 'Deep Learning', 'Dune'."
    else:
        return f"Database Query Executed for intent: '{query_intent}'. No specific data matched."
