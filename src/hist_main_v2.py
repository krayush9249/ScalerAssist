import os
import streamlit as st
from dotenv import load_dotenv
from hist_retriever import CustomPineconeRetriever
from hist_rag_chain_v2 import create_rag_chain

# Load environment variables
env_path = "/Users/kumarpersonal/Downloads/ScalerAssist/venv-scaler-assist/.env"
load_dotenv(dotenv_path=env_path)

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'ScalerAssist'

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INFER_MODEL_NAME = os.getenv("INFER_MODEL_NAME")

# Streamlit app config
st.set_page_config(page_title="ScalerAssist AI", layout="centered")

# Initialize session state for messages if not exists
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create retriever and RAG chain with hybrid memory (only once)
if "rag_chain" not in st.session_state:
    retriever = CustomPineconeRetriever()
    st.session_state.rag_chain = create_rag_chain(
        retriever=retriever,
        groq_api_key=GROQ_API_KEY,
        model=INFER_MODEL_NAME,
        window_size=4  
    )

# Header with New Chat button
col1, col2 = st.columns([3, 1])
with col1:
    st.title("ScalerAssist AI")
    st.markdown("Ask any question related to Scaler Academy.")

with col2:
    if st.button("New Chat", type="primary", use_container_width=True):
        # Clear the hybrid memory
        st.session_state.rag_chain.clear_memory()
        
        # Clear Streamlit session messages
        st.session_state.messages = []
        
        # Show success message briefly
        st.success("New chat started! Previous history cleared.")
        
        # Rerun to refresh the UI
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about Scaler Academy..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Fetching answer..."):
            try:
                result = st.session_state.rag_chain.invoke({
                    "question": prompt,
                    "chat_history": []  # This is ignored, hybrid memory handles it
                })
                answer = result["answer"]
                
                # Display the answer
                st.markdown(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                            
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})