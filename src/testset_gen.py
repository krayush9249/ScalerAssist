import os
from dotenv import load_dotenv
import pandas as pd
from text_splitter import semantic_split
from embeddings import BGEEmbedding

env_path = "/Users/kumarpersonal/Downloads/ScalerAssist/venv-scaler-assist/.env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INFER_MODEL_NAME = os.getenv("INFER_MODEL_NAME")

corpus_path = "/Users/kumarpersonal/Downloads/ScalerAssist/Context/cleaned_text.txt"
with open(corpus_path, "r", encoding="utf-8") as f:
    corpus = f.read()

docs = semantic_split(corpus)

from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=INFER_MODEL_NAME,
    temperature=0
)
embeddings = BGEEmbedding()

from ragas.llms import LangchainLLMWrapper
generator_llm = LangchainLLMWrapper(llm)

from ragas.embeddings import LangchainEmbeddingsWrapper
generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)

print(dataset.to_pandas())