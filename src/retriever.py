from langchain_core.documents import Document
from embeddings import BGEEmbedding
from pinecone_utils import get_pinecone_index

embed_func = BGEEmbedding()
index = get_pinecone_index()

def pinecone_retriever(question, k=5):
    query_vector = embed_func.embed_query(question)
    results = index.query(vector=query_vector, top_k=k, include_metadata=True)
    return [Document(page_content=match["metadata"]["text"]) for match in results["matches"]]
