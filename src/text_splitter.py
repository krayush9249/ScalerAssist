from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

def recursive_char_split(corpus, chunk_size=300, chunk_overlap=50):
    rc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    documents = rc_splitter.create_documents([corpus])

    for doc in documents:
        doc.metadata["chunk_text"] = doc.page_content
    return documents


def semantic_split(corpus):
    embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    
    sc_splitter = SemanticChunker(embed_model)
    document = Document(page_content=corpus)
    documents = sc_splitter.split_documents([document])

    for doc in documents:
        doc.metadata["chunk_text"] = doc.page_content
    return documents