import os
import langchain_google_genai

# Mock the entire ChatGoogleGenerativeAI class before it's imported in graph.py
class MockChatResponse:
    def __init__(self, content):
        self.content = content

class MockChatModel:
    def __init__(self, **kwargs):
        pass
    def invoke(self, prompt, **kwargs):
        if "classifier" in str(prompt).lower():
            return MockChatResponse("recommend_book")
        return MockChatResponse("Here are some book recommendations.")

langchain_google_genai.ChatGoogleGenerativeAI = MockChatModel

from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = "mock_key"
os.environ["SUPABASE_URL"] = "http://mock.supabase.co"
os.environ["SUPABASE_KEY"] = "mock_key"
os.environ["DATABASE_URL"] = "postgresql://mock_user:mock_pass@mock.supabase.co:5432/mock_db"

from langchain_core.messages import HumanMessage
import database
import graph

# Patch graph.py's internal ChatGoogleGenerativeAI reference
graph.ChatGoogleGenerativeAI = MockChatModel

def mock_get_supabase_client():
    return None
database.get_supabase_client = mock_get_supabase_client

def mock_get_similar_books(query_text, k=5):
    return "Result 1: A great sci-fi book."
graph.get_similar_books_from_supabase = mock_get_similar_books

def mock_query_database_tool(input_dict):
    return "Summary: Mock summary data."
database.query_database_tool = mock_query_database_tool

print("Testing Recommend Flow:")
initial_state = {
    "messages": [HumanMessage(content="I want to read a sci-fi book")],
    "user_id": "mock_user"
}
result = graph.app.invoke(initial_state)
print(result)

class MockChatModelSQL:
    def __init__(self, **kwargs):
        pass
    def invoke(self, prompt, **kwargs):
        if "classifier" in str(prompt).lower():
            return MockChatResponse("query_data")
        return MockChatResponse("Here are some book recommendations.")

graph.ChatGoogleGenerativeAI = MockChatModelSQL

print("Testing Query Flow:")
initial_state2 = {
    "messages": [HumanMessage(content="Summarize user data")],
    "user_id": "mock_user"
}
result2 = graph.app.invoke(initial_state2)
print(result2)
