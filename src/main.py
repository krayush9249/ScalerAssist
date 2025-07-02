import os
from dotenv import load_dotenv
from retriever import pinecone_retriever
from rag_chain import create_rag_chain
from langchain_core.runnables import RunnableLambda
import streamlit as st

# Load env variables
env_path = "/Users/kumarpersonal/Downloads/ScalerAssist/venv-scaler-assist/.env"
load_dotenv(dotenv_path=env_path)

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'ScalerAssist'

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INFER_MODEL_NAME = os.getenv("INFER_MODEL_NAME")

# Init retriever & rag chain
retriever_runnable = RunnableLambda(lambda x: pinecone_retriever(x["question"]))
rag_chain = create_rag_chain(retriever_runnable, GROQ_API_KEY, INFER_MODEL_NAME)

# Streamlit UI
st.set_page_config(page_title="ScalerAssist AI", layout="centered")
st.title("ScalerAssist AI")
st.markdown("Ask any question related to Scaler Academy.")

query = st.text_input("Enter your question here: ")

if st.button("Submit") and query.strip():
    with st.spinner("Fetching answer..."):
        response = rag_chain.invoke({"question": query})
    st.markdown("Answer")
    st.success(response)

