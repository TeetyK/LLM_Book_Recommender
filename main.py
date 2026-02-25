import gradio as gr
from dotenv import load_dotenv
import os
import sys

sys.path.append(os.path.abspath('src'))

try:
    from vector import get_retriever
except ImportError:
    from src.vector import get_retriever


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

print("--- Initializing Application ---")
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found. Please create a .env file with your key.")

vector_retriever = get_retriever()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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

recommender_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    verbose=False # Set to True to see the chain's internal steps
)

print("Recommendation chain created.")



def get_recommendation(message, history):

    print(f"User query: {message}")
    response = recommender_chain({"question": message})
    print(f"AI response: {response['answer']}")
    return response['answer']

def create_gradio_app():
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
    app.launch(share=False)
