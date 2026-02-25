import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import shutil

try:
    from src.preprocessing import preprocess_books_data
except ImportError:
    from preprocessing import preprocess_books_data


RAW_DATA_PATH = 'datasets/books.csv'
CLEANED_DATA_PATH = 'datasets/books_cleaned.csv'
VECTORSTORE_PATH = 'chroma_db_google' # Directory to save the vector store

def get_retriever(force_recreate: bool = False):

    run_preprocessing = False
    if not os.path.exists(CLEANED_DATA_PATH):
        run_preprocessing = True
        print(f"Cleaned data not found at {CLEANED_DATA_PATH}. Running preprocessing...")
    else:
        try:
            df_check = pd.read_csv(CLEANED_DATA_PATH, nrows=0) # Read only header
            if 'text_for_embedding' not in df_check.columns:
                print(f"Warning: Found '{CLEANED_DATA_PATH}', but it's missing the required 'text_for_embedding' column.")
                run_preprocessing = True
        except Exception as e:
            print(f"Warning: Could not validate '{CLEANED_DATA_PATH}'. It might be empty or corrupted. Error: {e}")
            run_preprocessing = True

    if run_preprocessing:
        print("Running data preprocessing to create a valid cleaned file...")
        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}. Please make sure it exists.")
        preprocess_books_data(RAW_DATA_PATH, CLEANED_DATA_PATH)
    else:
        print(f"Found valid cleaned data at {CLEANED_DATA_PATH}.")

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Please create a .env file with this key.")

    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")
    print("Initialized Google Generative AI Embeddings.")

    if force_recreate and os.path.exists(VECTORSTORE_PATH):
        print(f"Deleting existing vector store at {VECTORSTORE_PATH}...")
        shutil.rmtree(VECTORSTORE_PATH)

    if not os.path.exists(VECTORSTORE_PATH):
        print(f"Vector store not found at {VECTORSTORE_PATH}. Creating a new one...")
        
        books_df = pd.read_csv(CLEANED_DATA_PATH)
        books_df['text_for_embedding'] = books_df['text_for_embedding'].fillna('').astype(str)
        texts = books_df['text_for_embedding'].tolist()
        
        metadatas = books_df[['title', 'authors', 'categories']].to_dict('records')

        print(f"Creating vector store with {len(texts)} documents. This may take a while...")
        
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

    return db.as_retriever(search_kwargs={"k": 5})

if __name__ == '__main__':
    print("--- Running Vector Store Creation Script ---")
    retriever = get_retriever(force_recreate=False) 
    print("\n--- Script finished ---")
    print(f"Retriever created successfully. You can now use it for similarity searches.")
    
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

