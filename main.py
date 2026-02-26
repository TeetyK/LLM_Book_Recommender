
import streamlit as st
import pandas as pd
import os
import shutil
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- Constants ---
RAW_DATA_PATH = 'datasets/books.csv'
CLEANED_DATA_PATH = 'datasets/books_cleaned.csv'
VECTORSTORE_PATH = 'chroma_db_google'

# --- Data Preprocessing ---
def preprocess_books_data(input_path: str, output_path: str):
    """
    Loads book data, cleans it, combines relevant text fields,
    and saves the processed data to a new CSV file.
    """
    st.write(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path, on_bad_lines='warn')
    except Exception as e:
        st.error(f"Error reading CSV: {e}. Trying with 'python' engine.")
        df = pd.read_csv(input_path, engine='python', on_bad_lines='warn')

    columns_to_drop = [
        'isbn13', 'isbn10', 'subtitle', 'thumbnail', 'published_year',
        'average_rating', 'num_pages', 'ratings_count'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    text_cols = ['title', 'authors', 'categories', 'description']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')

    def create_embedding_text(row):
        return (f"Title: {row.get('title', '')}\n"
                f"Authors: {row.get('authors', '')}\n"
                f"Categories: {row.get('categories', '')}\n"
                f"Description: {row.get('description', '')}")

    df['text_for_embedding'] = df.apply(create_embedding_text, axis=1)
    df.dropna(subset=['title'], inplace=True)
    df = df[df['title'] != '']
    df.drop_duplicates(subset=['title', 'authors'], inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    st.write(f"Saving cleaned data to {output_path}...")
    df.to_csv(output_path, index=False)
    st.success("Preprocessing complete.")

# --- Vector Store and Retriever Initialization ---
@st.cache_resource
def get_retriever():
    """
    Initializes and returns a Chroma vector store retriever.
    - Checks for cleaned data, runs preprocessing if needed.
    - Checks for vector store, creates it if needed.
    """
    # 1. Check for and preprocess data if necessary
    if not os.path.exists(CLEANED_DATA_PATH):
        st.warning(f"Cleaned data not found. Running preprocessing...")
        if not os.path.exists(RAW_DATA_PATH):
            st.error(f"Raw data file not found at {RAW_DATA_PATH}. Cannot create vector store.")
            return None
        preprocess_books_data(RAW_DATA_PATH, CLEANED_DATA_PATH)

    # 2. Initialize Embeddings
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY not found. Please create a .env file with your key.", icon="🚨")
        st.stop()
        
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # 3. Create or Load Vector Store
    if not os.path.exists(VECTORSTORE_PATH):
        st.warning(f"Vector store not found. Creating a new one... This may take a moment.")
        
        books_df = pd.read_csv(CLEANED_DATA_PATH)
        books_df['text_for_embedding'] = books_df['text_for_embedding'].fillna('').astype(str)
        texts = books_df['text_for_embedding'].tolist()
        metadatas = books_df[['title', 'authors', 'categories']].to_dict('records')

        with st.spinner('Creating vector store... This is a one-time setup.'):
            db = Chroma.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas,
                persist_directory=VECTORSTORE_PATH
            )
        st.success("Vector store created and saved.")
    else:
        with st.spinner('Loading existing vector store...'):
            db = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)
        st.success("Vector store loaded.")

    return db.as_retriever(search_kwargs={"k": 5})

# --- LLM Chain Initialization ---
@st.cache_resource
def get_recommender_chain():
    """Initializes and returns the QA chain for book recommendations."""
    prompt_template = """You are an expert book recommender AI.
    A user will ask for book recommendations. Use the following pieces of context (retrieved book information) to answer the user's question.
    Your goal is to provide helpful and insightful recommendations based on the user's query and the provided book data.

    If the context is empty or you cannot find a suitable book, politely state that you couldn't find a matching book in the database and suggest they try a different query. Do not make up answers.
    For each recommended book, provide the title, author(s), and a brief, compelling explanation of why it fits the user's request.
    Present the recommendations in a clear, easy-to-read format (e.g., a numbered or bulleted list).

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:"""
    
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    retriever = get_retriever()
    if retriever is None:
        return None

    recommender_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    return recommender_chain


# --- Streamlit UI ---
st.set_page_config(page_title="📚 AI Book Recommender", layout="wide")

st.title("📚 AI Book Recommender")
st.markdown("""
Welcome! I'm your personal AI book expert. 
Ask for a recommendation based on a topic, genre, author, or even a book you've liked before.
""")

# Initialize chain
recommender_chain = get_recommender_chain()

if recommender_chain is None:
    st.error("Failed to initialize the recommendation engine. Please check the logs above.")
else:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("e.g., 'A classic sci-fi book about AI'"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Finding the perfect books for you..."):
                response = recommender_chain({"query": prompt})
                st.markdown(response["result"])
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["result"]})
