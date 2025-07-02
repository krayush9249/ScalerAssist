import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_core.documents import Document

env_path = "/Users/kumarpersonal/Downloads/ScalerAssist/venv-scaler-assist/.env"
load_dotenv(dotenv_path=env_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

def setup_pinecone(api_key, index_name):
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)

def upsert_documents(documents, embed_func):
    index = setup_pinecone(PINECONE_API_KEY, PINECONE_INDEX_NAME)
    texts = [doc.page_content for doc in documents]
    vectors = embed_func.embed_documents(texts)
    upsert_data = [
        {"id": f"doc-{i}", "values": vectors[i], "metadata": {"text": texts[i]}}
        for i in range(len(texts))
    ]
    index.upsert(vectors=upsert_data)

def get_pinecone_index():
    return setup_pinecone(PINECONE_API_KEY, PINECONE_INDEX_NAME)