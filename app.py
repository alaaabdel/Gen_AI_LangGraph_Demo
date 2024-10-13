from scripts.data_loader import init_cassio, load_documents, split_documents
from scripts.vectorstore_manager import VectorStoreManager
from scripts.query_router import QueryRouter
import os
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # To disable specific Hugging Face tokenizer warning
)


def main(query):
    # Initialize Cassandra
    init_cassio()

    # Load and split documents
    urls = [
        "https://developers.google.com/machine-learning/resources/prompt-eng",
        "https://huggingface.co/docs/transformers/main/en/tasks/visual_question_answering",
        "https://aws.amazon.com/what-is/retrieval-augmented-generation/",
    ]
    docs_list = load_documents(urls)
    doc_splits = split_documents(docs_list)

    # Initialize the vector store manager
    vector_store_manager = VectorStoreManager()
    vector_store_manager.add_documents(doc_splits)

    # Initialize query router with Groq API key and the vectorstore
    groq_api_key = os.getenv("groq_api_key")
    query_router = QueryRouter(groq_api_key, vector_store_manager)

    # Get the answer from the router
    answers = query_router.get_answer(query)
    breakpoint()

    # Print the answer
    if isinstance(answers, str):
        print(answers)
    else:
        print(answers[0].page_content)
        # for key,item in answer.metadata.items():
        #     print(key)
        #     print(item if key = "page_")

        # print(answer)


if __name__ == "__main__":
    # Use argparse to accept query from command line
    parser = argparse.ArgumentParser(description="Pass a query to the app.")
    parser.add_argument("--query", required=True, help="The query to process.")

    args = parser.parse_args()

    # Call the main function with the query
    main(args.query)
