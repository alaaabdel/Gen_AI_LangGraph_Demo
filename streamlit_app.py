import os
import warnings
import streamlit as st
from scripts.data_loader import init_cassio, load_documents, split_documents
from scripts.vectorstore_manager import VectorStoreManager
from scripts.query_router import QueryRouter

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # Disable specific Hugging Face tokenizer warning
)


def main():
    # Create a title for the app
    st.title("Document Query App")

    # Initialize Cassandra
    init_cassio()

    # Load and split documents
    urls = [
        "https://developers.google.com/machine-learning/resources/prompt-eng",
        "https://huggingface.co/docs/transformers/main/en/tasks/visual_question_answering",
        "https://aws.amazon.com/what-is/retrieval-augmented-generation/",
    ]

    # Load documents and split them into chunks
    docs_list = load_documents(urls)
    doc_splits = split_documents(docs_list)

    # Initialize the vector store manager
    vector_store_manager = VectorStoreManager()
    vector_store_manager.add_documents(doc_splits)

    # Initialize query router with Groq API key and the vector store
    groq_api_key = os.getenv("groq_api_key")
    query_router = QueryRouter(groq_api_key, vector_store_manager)

    # Create an input field for user queries
    query = st.text_input("Enter your query:", value="", key="query")
    # Process the query when the user submits it
    if st.button("Submit"):
        if query:
            # Get the answer from the router
            answers = query_router.get_answer(query)

            if isinstance(answers, str):
                answer = answers
            else:
                # Print the answer
                answer = answers[0].page_content

            # Display the answer
            st.subheader("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a query.")


if __name__ == "__main__":
    main()
