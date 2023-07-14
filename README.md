# chromadb_qa_bot

Ask PDF a question using GPT-3.5-turbo, ChromaDB and LangChain.

2023-07-14, Johannes Köppern

## Project Description

- Use of LangChain
- Load a PDF file (also included in this repo)
- Do word embedding via OpenAI Embedding API
- Store result in ChromaDB and persist it
- Antlerntive: Load existing ChromaDB
- Ask the PDF a question using GPT-3.5-turbo

## Table of Contents

Include a table of contents to help users navigate through the README.
See also this [blog post](https://betterofjohn.com/uncategorized/custom-question-answering-qa-bot-transforming-pdf-interactions-with-langchain-and-chromadb/).

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

- Create and activate Python enironment, e.g. via `conda create -n chromadb_qa_bot python=3.9` and `conda activate chromadb_qa_bot`
- Install requirements, e.g. via `pip install -r requirements.txt`

## Usage

The app which creates a vecotr store from a PDF and then allows a conversation with it can be ffound in `app/app.py`:

### Parameters

- *create_embeding*: If True the vecotor db is created based on the PDF's content. Otherwied it's loaded from the persisted one.
- *filename_pdf*: Defines which PDF is consided to create the vector db.
- *persist_directory*: Defines in which fileder the vector db is persisted in.

``` python
# -------------------------------------------------------
# Parameters
# -------------------------------------------------------
create_embeding = False

filename_pdf = "docs/olivia_rodrigo.pdf"

persist_directory = 'db'
```

### Functions

```python
# -------------------------------------------------------
# Functions
# -------------------------------------------------------
def create_retriever(vectordb)
def provide_vector_db(create_embeding)
def import_api_key()
```

### Main app

1. The OpenAI API key is loaded from a text file outside of this repository and stored in the environment variable `OPENAI_API_KEY`.
2. The Chroma vector db is created/loaded.
3. Questions are asked using GPT-3.5-turbo and named vector db.


## License

Apache License, Version 2.0

## Contact information
[My website](https://betterofjohn.com/)