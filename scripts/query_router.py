import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

from typing import Literal


# Data model for routing
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ..., description="Choose between Wikipedia or vectorstore."
    )


class QueryRouter:
    def __init__(self, groq_api_key, vectorstore):
        """
        Initialize the query router.
        Args:
            groq_api_key (str): API key for Groq LLM.
            vectorstore: The vectorstore instance to use for document retrieval.
        """
        self.vectorstore = vectorstore
        self.llm_router = self.create_router(groq_api_key)
        self.route_prompt = self.create_route_prompt()

    def create_router(self, groq_api_key):
        """
        Create the LLM-based router.
        Args:
            groq_api_key (str): API key for the Groq LLM service.
        Returns:
            ChatGroq: An instance of the structured LLM router.
        """
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")
        structured_llm_router = llm.with_structured_output(RouteQuery)
        return structured_llm_router

    def create_route_prompt(self):
        """
        Create a system prompt to guide the LLM-based router.
        Returns:
            ChatPromptTemplate: A prompt template for routing questions.
        """
        system_prompt = """You are an AI assistant specialized in directing user queries to the most appropriate information source. You have access to two primary resources:
                    A specialized vectorstore containing in-depth information on:
                    1. Prompt engineering
                    2. LLM Hallucination
                    3. adversial attacks
                    A general Wikipedia search function for broader topics
                    Your task is to analyze each user question and determine the optimal source for answering:
                    For queries related to the four specific topics listed above, direct the user to the vectorstore for accurate, specialized information.
                    For all other questions or topics outside these areas, utilize the Wikipedia search function to provide general knowledge and background information.
                    Ensure you make a clear decision for each query, selecting the most appropriate resource to deliver the most relevant and accurate response to the user."""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )

    def route_question(self, query):
        """
        Route a user question to the most relevant data source (vectorstore or Wikipedia).
        Args:
            query (str): The user's query.
        Returns:
            str: Next step - either vectorstore or wiki_search.
        """
        result = self.route_prompt | self.llm_router
        source = result.invoke({"question": query})

        return source.datasource

    def retrieve_from_vectorstore(self, query):
        """
        Retrieve relevant documents from the vectorstore.
        Args:
            query (str): The user's query.
        Returns:
            list: A list of relevant documents.
        """
        return self.vectorstore.retrieve_documents(query)

    def retrieve_from_wikipedia(self, query):
        """
        Retrieve relevant content from Wikipedia based on the query.
        Args:
            query (str): The user's query.
        Returns:
            str: Wikipedia article summary.
        """
        wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
        wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

        wiki_results = wiki_tool.invoke({"query": query})
        return wiki_results or "No results found."

    def get_answer(self, query):
        """
        Get the answer to a query by routing it to the appropriate source (Wikipedia or vectorstore).
        Args:
            query (str): The user's query.
        Returns:
            str: The retrieved answer (from Wikipedia or vectorstore).
        """
        data_source = self.route_question(query)
        if data_source == "vectorstore":
            return self.retrieve_from_vectorstore(query)
        elif data_source == "wiki_search":
            return self.retrieve_from_wikipedia(query)
        else:
            return "Could not route the question."
