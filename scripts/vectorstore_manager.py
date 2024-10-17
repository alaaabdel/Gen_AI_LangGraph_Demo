from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper


class VectorStoreManager:
    def __init__(
        self, model_name="all-MiniLM-L6-v2", device="cpu", table_name="gen_ai_table"
    ):
        """
        Initialize the vector store manager with embedding and vectorstore configurations.
        Args:
            model_name (str): Name of the HuggingFace model to use for embeddings.
            device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
            table_name (str): Name of the Cassandra table to store vectors.
        """
        self.model_name = model_name
        self.device = device
        self.table_name = table_name
        self.embedding_model = self.create_embedding_model()
        self.vectorstore = self.setup_vectorstore()

    def create_embedding_model(self):
        """
        Create an embedding model using HuggingFace Bge embeddings.
        Returns:
            HuggingFaceBgeEmbeddings: The embedding model.
        """
        model_kwargs = {"device": self.device}
        encode_kwargs = {"normalize_embeddings": True}
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        return embedding_model

    def setup_vectorstore(self):
        """
        Set up the Cassandra vector store with the specified embedding model.
        Returns:
            Cassandra: The initialized Cassandra vector store.
        """
        return Cassandra(embedding=self.embedding_model, table_name=self.table_name)

    def add_documents(self, documents):
        """
        Add documents to the Cassandra vector store.
        Args:
            documents (list): List of document splits to be added to the vector store.
        """
        self.vectorstore.add_documents(documents)
        print(f"Inserted {len(documents)} documents into the vector store.")

    def retrieve_documents(self, query):
        """
        Retrieve documents from the vector store based on a query.
        Args:
            query (str): The query to retrieve relevant documents for.
        Returns:
            list: List of relevant documents retrieved from the vector store.
        """
        return self.vectorstore.similarity_search(query, k=1)
