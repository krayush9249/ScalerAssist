from text_splitter import semantic_split
from embeddings import BGEEmbedding
from pinecone_utils import upsert_documents

corpus_path = "/Users/kumarpersonal/Downloads/ScalerAssist/Context/cleaned_text.txt"

with open(corpus_path, "r", encoding="utf-8") as f:
    corpus = f.read()

docs = semantic_split(corpus)
embed_func = BGEEmbedding()

upsert_documents(docs, embed_func)