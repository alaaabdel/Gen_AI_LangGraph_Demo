# GEN AI Demo for document query 

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
