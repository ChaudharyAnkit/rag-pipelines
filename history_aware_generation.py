from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

persist_directory = "dv/chroma_db"

# load embeddings and vector store
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small", dimensions=1024)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

retriever = db.as_retriever(search_kwargs={"k": 5})

model = ChatOpenAI(model="gpt-4o")
chat_history = []


def ask_question(query):
    global chat_history

    if chat_history:
        rewrite_messages = [
            SystemMessage(
                content=(
                    "Given the chat history, rewrite the new question to be standalone and searchable. "
                    "Only rewrite the question, do not answer it."
                )
            )
        ] + chat_history + [HumanMessage(content=f"new question: {query}")]

        rewrite_response = model.invoke(rewrite_messages)
        search_query = rewrite_response.content.strip()
        print(f"Rewritten Search Query: {search_query}")
    else:
        search_query = query

    relevant_docs = retriever.invoke(search_query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    final_query = f"""Based on the following documents, please answer this question: {query}

Documents:
{context}

Please provide a clear, helpful answer using only the information provided. If you can't find the answer in the document, say "I don't have enough information about the answer based on provided document"."""

    answer_messages = [
        SystemMessage(
            content="You are a helpful assistant that answers questions based on the provided documents."),
        HumanMessage(content=final_query),
    ]
    answer_response = model.invoke(answer_messages)

    print("---Generated Response---")
    print(answer_response.content)

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer_response.content))

    return answer_response


if __name__ == "__main__":
    while True:
        user_query = input(
            "\nEnter your question (or 'exit' to quit): ").strip()
        if user_query.lower() == 'exit':
            break
        if user_query:
            print(f"User Query: {user_query}")
            ask_question(user_query)
