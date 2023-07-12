# This application let's the user chat with a PDF document.
# LangChain is used to load a PDF document, to create emedding, store them
# in a ChromaDB and chat with this document using an OpenAI LLM.

# As LLM OpenAI's GPT-3.5 is used.
# OpenAI's Emmedding function creates emendding for named PDF.
# The loaded PDF is a print-out of the Wikipedia article on Olivia Rodrigo

# 2023-07-11, J. Köppern

import openai
import os
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader



# -------------------------------------------------------
# Parameters
# -------------------------------------------------------
create_embeding = False

filename_pdf = "docs/olivia_rodrigo.pdf"

persist_directory = 'db'

# -------------------------------------------------------
# Functions
# -------------------------------------------------------
def square_input(input):
    return input**2

def create_retriever(vectordb):
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), 
        chain_type="stuff", 
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
    )

    return qa

def provide_vector_db(create_embeding):
    embeddings = OpenAIEmbeddings()


    if create_embeding:
        print("Create vectordb")
              
        # PDF loader
        loader = PyPDFLoader(filename_pdf)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, 
            chunk_overlap=20)


        texts = splitter.split_documents(
            documents
        )


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

def import_api_key():
    load_dotenv(
        find_dotenv(),
    override=True
)
    
# -------------------------------------------------------
# Main section of the script
# -------------------------------------------------------
if __name__ == '__main__':
    import_api_key()

    vectordb = provide_vector_db(
        create_embeding=create_embeding
    )

    # QA
    qa = create_retriever(vectordb=vectordb)

    queries = ["Who is Olivia Rodrigo?", "Who is Albert Einstein?"]

    for query in queries:
        result = qa({"query": query})

        print(result["result"])

        print(result["source_documents"])

