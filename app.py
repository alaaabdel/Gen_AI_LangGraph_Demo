from scripts.data_loader import init_cassio, load_documents, split_documents
from scripts.vectorstore_manager import VectorStoreManager
from scripts.query_router import QueryRouter
import os
import argparse
import warnings
warnings.filterwarnings("ignore") 
os.environ["TOKENIZERS_PARALLELISM"] = "false" #To disable specific Hugging Face tokenizer warning

def main(query):
    # Initialize Cassandra
    init_cassio()

    # Load and split documents
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        "https://mlabonne.github.io/blog/posts/A_Beginners_Guide_to_LLM_Finetuning.html"
    ]
    docs_list = load_documents(urls)
    doc_splits = split_documents(docs_list)

    # Initialize the vector store manager
    vector_store_manager = VectorStoreManager()
    vector_store_manager.add_documents(doc_splits)

    # Initialize query router with Groq API key and the vectorstore
    groq_api_key = os.getenv('groq_api_key')
    query_router = QueryRouter(groq_api_key, vector_store_manager)

    # Get the answer from the router
    answer = query_router.get_answer(query)

    # Print the answer
    print("*"*40+answer)

if __name__ == "__main__":
    # Use argparse to accept query from command line
    parser = argparse.ArgumentParser(description="Pass a query to the app.")
    parser.add_argument("--query", required=True, help="The query to process.")

    args = parser.parse_args()

    # Call the main function with the query
    main(args.query)