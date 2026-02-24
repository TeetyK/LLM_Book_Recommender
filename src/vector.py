from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pandas as pd
def retrieve_semantic_recommandations(
    books:pd.DataFrame,
    raw_documents:str,
    query: str,
    top_k: int = 10,
) -> pd.DataFrame:
    text_splitter = CharacterTextSplitter(chunk_size=0,chunk_overlap=0,separator="\n")
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(
        documents,
        embedding=OpenAIEmbeddings
    )
    recs = db_books.similarity_search(query,top_k)
    books_list = []
    for i in range(0, len(recs)):
        books_list += [int(recs[i].page_content.strip("").split()[0])]
    return books[books["isbn13"].isin(books_list)] 