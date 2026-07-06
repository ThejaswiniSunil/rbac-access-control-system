# ========== CONFIG ==========
from pathlib import Path
import os
import pandas as pd
from collections import defaultdict
from langchain_core.documents import Document
import sqlite3

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_cohere import CohereRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from .secret_key import langchain_key, cohere_api_key, groq_api_key



os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "RAG"
os.environ["LANGCHAIN_API_KEY"] = langchain_key
os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["COHERE_API_KEY"] = cohere_api_key


openai_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="my_collection",
    persist_directory="chroma_db",
    embedding_function=openai_embeddings
)


def embed_documents_to_vectorstore(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore.add_documents(splits)

    print("Documents embedded and saved to vectorstore.")
    print("Total documents:", len(vectorstore.get()["documents"]))


def load_file(filepath, role):
    ext = Path(filepath).suffix.lower()
    try:
        if ext == ".csv":
            df1 = pd.read_csv(filepath)
            documents = []
            for row in df1.to_dict(orient="records"):
                content = "\n".join(f"{k}: {v}" for k, v in row.items())
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"role": role.lower(), "source": Path(filepath).name}
                    )
                )
            return documents

        elif ext == ".md":
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return [
                Document(
                    page_content=content,
                    metadata={"role": role.lower(), "source": Path(filepath).name}
                )
            ]
        else:
            return None

    except Exception as e:
        print(f"Failed to process {filepath}: {e}")
        return None


def run_indexer():
    conn = sqlite3.connect("roles_docs.db")
    c = conn.cursor()
    c.execute("SELECT id, filepath, role FROM documents WHERE embedded = 0")

    all_docs = []

    for doc_id, path, role in c.fetchall():
        docs = load_file(path, role)
        if docs:
            if isinstance(docs, list):
                all_docs.extend(docs)
            else:
                all_docs.append(docs)

            c.execute("UPDATE documents SET embedded = 1 WHERE id = ?", (doc_id,))

    if all_docs:
        embed_documents_to_vectorstore(all_docs)
        conn.commit()

    conn.close()
    print(f"Indexed {len(all_docs)} document chunks.")


system_prompt = (
    "You are an assistant for summarizing and answering queries from internal company documents.\n"
    "Always use the retrieved context to answer the query, even if partial.\n"
    "Do not guess. If data is not found, explain what you searched for.\n"
    "When responding:\n"
    "- Add **Source** from document metadata if possible.\n"
    "- Use headers\n"
    "- Use bullet points\n"
    "- For CSV-style data, format in table with two columns\n"
    "\n{context}"
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

question_answering_chain = create_stuff_documents_chain(model, chat_prompt)

def wrap_with_reranker(retriever, cohere_api_key, top_n=4):
    reranker = CohereRerank(cohere_api_key=cohere_api_key, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=retriever
    )

def get_rag_chain(user_role: str, cohere_api_key: str = None):
    user_role = user_role.lower()

    if user_role == "c-level":
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    elif user_role == "general":
        retriever = vectorstore.as_retriever(search_kwargs={
            "k": 4,
            "filter": {"role": "general"}
        })

    else:
        retriever = vectorstore.as_retriever(search_kwargs={
            "k": 4,
            "filter": {
                "role": {"$in": [user_role, "general"]}
            }
        })

    if cohere_api_key:
        print("Using cohere reranker")
        retriever = wrap_with_reranker(retriever, cohere_api_key)

    return create_retrieval_chain(retriever, question_answering_chain)
