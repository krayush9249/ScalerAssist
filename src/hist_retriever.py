from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Any
from embeddings import BGEEmbedding
from pinecone_utils import get_pinecone_index

class CustomPineconeRetriever(BaseRetriever):
    k: int = 5  # Declare as class field with default value
    embed_func: BGEEmbedding = None  # Declare as class field
    index: Any = None  # Declare as class field with proper type annotation
    
    def __init__(self, k: int = 5):
        super().__init__(k=k)  # Pass k to parent constructor
        self.embed_func = BGEEmbedding()
        self.index = get_pinecone_index()

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.embed_func.embed_query(query)
        results = self.index.query(vector=query_vector, top_k=self.k, include_metadata=True)
        return [
            Document(page_content=match["metadata"]["text"], metadata=match["metadata"])
            for match in results["matches"]
        ]