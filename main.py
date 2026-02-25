import gradio as gr
from dotenv import load_dotenv
import os
import sys

# Add src to path to allow imports from other files in the src directory
sys.path.append(os.path.abspath('src'))

# Import our vector store creator and retriever
try:
    from vector import get_retriever
except ImportError:
    # Handle the case where the script is run from the root directory
    from src.vector import get_retriever


# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- 1. Load Environment Variables and Setup ---
print("--- Initializing Application ---")
# Load API keys from .env file
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    # Display error in Gradio if the key is missing
    raise ValueError("OPENAI_API_KEY not found. Please create a .env file with your key.")

print("Loading the vector store and retriever... (This might take a moment on first launch)")
# This will create the store if it doesn't exist, or load it if it does.
vector_retriever = get_retriever()
print("Retriever loaded successfully.")

# --- 2. Set up the LangChain Recommendation Chain ---

# Initialize the LLM
# We use gpt-3.5-turbo as it's fast and cost-effective.
# A non-zero temperature adds a bit of creativity to the responses.
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

# We need to keep track of the conversation history for follow-up questions
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the prompt template
# This is how we instruct the AI on how to behave.
prompt_template = """You are an expert book recommender AI. A user will ask for book recommendations.
Use the following pieces of context (retrieved book information) to answer the user's question.
Your goal is to provide helpful and insightful recommendations based on the user's query and the provided book data.

If you don't know the answer from the context, or if the context is empty, just say that you couldn't find a matching book in your database and suggest they try a different query, don't try to make up an answer.
For each recommended book, provide the title, author(s), and a brief, compelling explanation of why it fits the user's request.
Present the recommendations in a clear, easy-to-read format (e.g., a numbered or bulleted list).

Context:
{context}

Question:
{question}

Helpful Answer:"""

QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create the Conversational Retrieval Chain
# This chain combines the retriever, the memory, and the LLM.
recommender_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    verbose=False # Set to True to see the chain's internal steps
)

print("Recommendation chain created.")


# --- 3. Define the Gradio Chatbot Logic ---

def get_recommendation(message, history):
    """
    This function is called by the Gradio interface for each user message.
    It takes the user's message and the chat history, sends it to the chain,
    and returns the AI's response.
    """
    print(f"User query: {message}")
    # The chain takes care of managing the history and calling the LLM
    response = recommender_chain({"question": message})
    print(f"AI response: {response['answer']}")
    return response['answer']

# --- 4. Build and Launch the Gradio UI ---

def create_gradio_app():
    """Creates and returns the Gradio app blocks."""
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), title="AI Book Recommender") as app:
        gr.Markdown(
            """
            <div style="text-align: center;">
                <h1>📚 AI Book Recommender</h1>
                <p>Ask for a book recommendation based on a topic, genre, or another book you like!</p>
            </div>
            """
        )

        gr.ChatInterface(
            fn=get_recommendation,
            examples=[
                "Suggest a classic science fiction book about artificial intelligence.",
                "I'm looking for a fantasy book with a unique magic system.",
                "What are some good books on the history of computing?",
                "I liked 'Project Hail Mary', can you recommend something similar?"
            ],
            chatbot=gr.Chatbot(height=500, label="Chat"),
            textbox=gr.Textbox(placeholder="e.g., 'Recommend a thriller with a surprising twist.'", container=False, scale=7),
            title=None,
            description=None,
            submit_btn="▶️ Send",
            retry_btn="🔄 Regenerate",
            undo_btn="↩️ Undo",
            clear_btn="🗑️ Clear",
        )
    return app

if __name__ == "__main__":
    print("Starting Gradio application...")
    app = create_gradio_app()
    # share=False keeps the app local. Set to True to create a public link.
    app.launch(share=False)
