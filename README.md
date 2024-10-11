# GEN AI Demo for document query 

Query-Based Document Retrieval App

A Streamlit app that enables users to query and retrieve relevant information from a set of documents. The app leverages HuggingFace embeddings and stores them in a Cassandra vector database for efficient document retrieval. Key techniques include document splitting, embedding generation, and retrieval-based NLP using vector search.



## Setup dependencies

- create a table in Astra DB
- create token for groq API to access llama model
- store env variables for both astra DB and groq 

2. Now run this command to install dependenies in the `pyproject.toml` file. 

```python
make create-env
```

```python
make install-env
```

## To run the app in command line 

```python
python app.py --query <enter your question>
```


## To run the app on streamlit

```python
streamlit run streamlit_app.py
```
