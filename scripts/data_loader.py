import os
import cassio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv


def init_cassio():
    """
    Initialize the connection to the Cassandra database using environment variables.
    """
    load_dotenv()  # Load environment variables from .env file
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("db_token")
    ASTRA_DB_ID = os.getenv("db_id")

    if not ASTRA_DB_APPLICATION_TOKEN or not ASTRA_DB_ID:
        raise ValueError(
            "Cassandra DB token or ID is missing from environment variables."
        )

    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
    print("Cassandra connection initialized.")


def load_documents(urls):
    """
    Load documents from a list of URLs.

    Args:
        urls (list): List of URLs to load documents from.

    Returns:
        list: List of documents loaded from the URLs.
    """
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list


def split_documents(docs_list, chunk_size=200, chunk_overlap=10):
    """
    Split documents into chunks for vectorization.

    Args:
        docs_list (list): List of documents to split.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap size between chunks.

    Returns:
        list: List of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits
