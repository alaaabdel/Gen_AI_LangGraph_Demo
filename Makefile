.ONESHELL:
SHELL =/bin/bash

ENV_NAME = gen_ai_demo1
PYENV_ACTIVATE = source $(ENV_NAME)/bin/activate

create-env:
	python3 -m venv $(ENV_NAME)

install-env:
	$(PYENV_ACTIVATE) && pip install poetry
	poetry lock
	poetry install --no-root --all-extras
	#poetry run pre-commit install
	poetry add ipykernel 
	poetry run python -m ipykernel install --user --name $(ENV_NAME)