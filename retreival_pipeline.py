from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
persist_directory = "dv/chroma_db"
persistent_vector_store = Chroma(persist_directory=persist_directory)

# load embeddings and vector store
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small", dimensions=1024)

db = Chroma(persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
            )

# search for the relevant documents
query = "How many years of experience do Anki Choudhary have in the company rosemerta?"

# retrieve relevant documents
retriever = db.as_retriever(
    search_kwargs={"k": 5}  # returns the top 5 most similar documents
)

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # only returns documents with a similarity score above 0.3
#     }
# )

relevant_docs = retriever.invoke(query)
print(f"User Query: {query}")
# display results
# print("---Context----")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n {doc.page_content}\n")

# combine the query and the relevant documents
context = "\n".join([doc.page_content for doc in relevant_docs])
final_query = f"""Based on the following document, please answer this question: {query}

Documents:
{context}

Please provide a clear, helpful answer using only the information provided. If you can't find the answer in the document, say "I don't have enough information about the answer based on provided document"."""

# create a Chat OpenAI instance
model = ChatOpenAI(model="gpt-4o")

# define the message for the chat

messages = [
    SystemMessage(
        content="You are a helpful assistant that answers questions based on the provided documents."),
    HumanMessage(content=final_query)
]

# invoke the chat model
response = model.invoke(messages)

# display the response
print("---Genrated Response---")
print(response.content)
