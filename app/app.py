# This application let's the user chat with a PDF document.
# LangChain is used to load a PDF document, to create emedding, store them
# in a ChromaDB and chat with this document using an OpenAI LLM.

# As LLM OpenAI's GPT-3.5 is used.
# OpenAI's Emmedding function creates emendding for named PDF.
# The loaded PDF is a print-out of the Wikipedia article on Olivia Rodrigo

# 2023-07-11, J. Köppern

import openai
import os
import sys
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import JSONLoader, PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader


# -------------------------------------------------------
# Parameters
# -------------------------------------------------------
create_embedding = False

filename_pdf = "recipes.json"

persist_directory = 'db'

# -------------------------------------------------------
# Functions
# -------------------------------------------------------
def create_retriever_and_answerer(vectordb):
    """
    Creates a retriever and answerer object for the given vector database.

    Args:
        vectordb (Chroma): The vector database.

    Returns:
        RetrievalQA: The retriever and answerer object.
    """
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        verbose=True
    )

    return qa

def provide_vector_db(create_embedding):
    """
    Provides the vector database.

    Args:
        create_embedding (bool): Whether to create the embedding or load an existing one.

    Returns:
        Chroma: The vector database.
    """
    embeddings = OpenAIEmbeddings()

    if create_embedding:
        print("Create vectordb")

        try:
            loader = JSONLoader(filename_pdf, jq_schema='.', text_content=False)
        except FileNotFoundError:
            print(f"File {filename_pdf} not found.")
            sys.exit(1)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20)

        texts = splitter.split_documents(documents)

        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory
        )

        vectordb.persist()
    else:
        print("Load vectordb")

        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    return vectordb

def load_api_key():
    """
    Loads the API key from the environment variables.
    """
    load_dotenv(find_dotenv(), override=True)

def main_script(create_embedding, create_retriever_and_answerer, provide_vector_db, load_api_key):
    """
    The main script that executes the chat with the PDF document.

    Args:
        create_embedding (bool): Whether to create the embedding or load an existing one.
        create_retriever_and_answerer (function): Function to create the retriever and answerer object.
        provide_vector_db (function): Function to provide the vector database.
        load_api_key (function): Function to load the API key.
    """
    load_api_key()

    vectordb = provide_vector_db(create_embedding=create_embedding)

    retriever_and_answerer = create_retriever_and_answerer(vectordb=vectordb)

    queries = ["Welche Rezepte enthalten Bohnen?"]

    for query in queries:
        result = retriever_and_answerer({"query": query})

        print(result["result"])
        print(result["source_documents"])

# -------------------------------------------------------
# Main section of the script
# -------------------------------------------------------
if __name__ == '__main__':
    main_script(
        create_embedding,
        create_retriever_and_answerer,
        provide_vector_db,
        load_api_key)
