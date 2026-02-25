import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import shutil

# Import the preprocessing function from the other src file
try:
    from src.preprocessing import preprocess_books_data
except ImportError:
    from preprocessing import preprocess_books_data


# --- Constants ---
# It's good practice to define paths and constants at the top
RAW_DATA_PATH = 'datasets/books.csv'
CLEANED_DATA_PATH = 'datasets/books_cleaned.csv'
VECTORSTORE_PATH = 'chroma_db_openai' # Directory to save the vector store

def get_retriever(force_recreate: bool = False):
    """
    Creates or loads a Chroma vector store and returns a retriever.

    This function encapsulates the entire data pipeline:
    1. Checks for cleaned data, runs preprocessing if needed.
    2. Checks for a persisted vector store.
    3. Creates the store from data if it doesn't exist or if force_recreate is True.
    4. Loads the store if it exists.
    5. Returns a retriever object for similarity searches.

    Args:
        force_recreate (bool): If True, will delete and recreate the vector store.

    Returns:
        A Chroma retriever object.
    """
    # --- Step 1: Ensure cleaned data is available ---
    if not os.path.exists(CLEANED_DATA_PATH):
        print(f"Cleaned data not found at {CLEANED_DATA_PATH}. Running preprocessing...")
        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}. Please make sure it exists.")
        preprocess_books_data(RAW_DATA_PATH, CLEANED_DATA_PATH)
    else:
        print(f"Found cleaned data at {CLEANED_DATA_PATH}.")

    # --- Step 2: Load Environment Variables ---
    # This will load the OPENAI_API_KEY from a .env file
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please create a .env file with this key.")

    # --- Step 3: Initialize Embeddings ---
    # Using a powerful and cost-effective OpenAI embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Initialized OpenAI Embeddings.")

    # --- Step 4: Create or Load Vector Store ---
    if force_recreate and os.path.exists(VECTORSTORE_PATH):
        print(f"Deleting existing vector store at {VECTORSTORE_PATH}...")
        shutil.rmtree(VECTORSTORE_PATH)

    if not os.path.exists(VECTORSTORE_PATH):
        print(f"Vector store not found at {VECTORSTORE_PATH}. Creating a new one...")
        
        # Load the cleaned data
        books_df = pd.read_csv(CLEANED_DATA_PATH)
        # Ensure the text column is not null and is of type string
        books_df['text_for_embedding'] = books_df['text_for_embedding'].fillna('').astype(str)
        texts = books_df['text_for_embedding'].tolist()
        
        # Create metadata to store useful info alongside the vectors
        metadatas = books_df[['title', 'authors', 'categories']].to_dict('records')

        print(f"Creating vector store with {len(texts)} documents. This may take a while...")
        
        # Create the vector store from documents
        db = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=VECTORSTORE_PATH
        )
        print("Vector store created and persisted.")

    else:
        print(f"Loading existing vector store from {VECTORSTORE_PATH}.")
        db = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)

    # Return the database as a retriever
    # 'k=5' means it will retrieve the top 5 most similar documents
    return db.as_retriever(search_kwargs={"k": 5})

if __name__ == '__main__':
    print("--- Running Vector Store Creation Script ---")
    # This will create the vector store from scratch if it doesn't exist.
    # To force recreation, set force_recreate=True.
    # Note: Creating embeddings for the first time requires an internet connection
    # and your OpenAI API key, and may incur costs.
    retriever = get_retriever(force_recreate=False) 
    print("\n--- Script finished ---")
    print(f"Retriever created successfully. You can now use it for similarity searches.")
    
    # Example usage:
    print("\n--- Example Search ---")
    try:
        query = "A book about space travel and philosophy"
        results = retriever.get_relevant_documents(query)
        print(f"Top 5 results for the query: '{query}'")
        if results:
            for doc in results:
                print(f"- Title: {doc.metadata.get('title', 'N/A')}")
        else:
            print("No relevant documents found.")
    except Exception as e:
        print(f"An error occurred during the example search: {e}")
        print("This might be due to API key issues or network problems.")

