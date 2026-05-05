import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

# method to load web pages


def load_web_documents(urls):
    print("Loading documents from web URLs...")

    loader = WebBaseLoader(urls)
    documents = loader.load()

    print(f"Loaded {len(documents)} web documents.")
    return documents

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


def create_document_chunks(documents, chunk_size=500):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size)
    document_chunks = text_splitter.split_documents(documents)
    return document_chunks


def create_vector_store(document_chunks, persist_directory="dv/chroma_db"):
    print("Creating vector store...")
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small", dimensions=1024)
    vector_store = Chroma.from_documents(
        documents=document_chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("Vector store created.")
    return vector_store


def main():
    print("Starting the ingestion pipeline...")
    # option 1 load documents from local txt files
    documents = load_documents()
    # option 2 load documents from web urls

    urls = [
        "https://www.shorturl.at/terms-of-service.php"
    ]
    web_docs = load_web_documents(urls)
    # combine both documents
    if documents and web_docs:
        documents.extend(web_docs)

    if documents:
        document_chunks = create_document_chunks(documents)
        print(f"Created {len(document_chunks)} document chunks.")
        for ch in document_chunks:
            # print the first 100 characters of each chunk
            print(ch.page_content[:100])
        vector_store = create_vector_store(document_chunks)
        print("Ingestion pipeline completed.")


if __name__ == "__main__":
    main()
