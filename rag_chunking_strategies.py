import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from ingestion_pipeline import create_vector_store

load_dotenv()


# method to load the txt files


def load_documents(docspath="docs"):
    print(f"Loading documents from {docspath}...")

    # load only if directory found
    if os.path.exists(docspath):
        directory_loader = DirectoryLoader(
            docspath, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        documents = directory_loader.load()
        return documents
    else:
        print(f"No documents found in {docspath}.")

# create chunks from loaded documents


# def create_document_chunks(documents, chunk_size=500):
#     text_splitter = CharacterTextSplitter(
#         chunk_size=chunk_size, chunk_overlap=0, separator="\n\n")
#     document_chunks = text_splitter.split_documents(documents)
#     return document_chunks


def create_document_chunks(documents, chunk_size=500):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, separators=["\n\n", "\n", " ", ""])
    document_chunks = text_splitter.split_documents(documents)
    return document_chunks


def Main():
    documents = load_documents()
    document_chunks = create_document_chunks(
        documents)

    for i, ch in enumerate(document_chunks, start=1):
        # print the chunking conent and number of characters in the chunk with serial numbers in the chunks
        print(
            f"---------------------Chunk {i}: Number of characters: {len(ch.page_content)}------------------")
        print(f"Chunk content: {ch.page_content}\n\n")


if __name__ == "__main__":
    Main()
