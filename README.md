# 🤖 ScalerAssist AI — RAG-based Chatbot

**A Retrieval-Augmented Generation (RAG) assistant to explore Scaler Academy's offerings, built using Pinecone, OpenAI/Groq LLMs, and Streamlit.**

---

## 📌 Overview

ScalerAssist AI is a conversational agent powered by Retrieval-Augmented Generation. It uses Scaler's internal course brochures and learning documents to help users query:

* Courses and Curriculum (Beginner, Intermediate, Advanced)
* Placement outcomes and statistics
* Specializations and Career Tracks
* Instructor and mentor background
* Alumni success stories

---

## 🧠 Features

* 🔍 **Contextual Retrieval** with Pinecone vector store
* 🧠 **Hybrid Memory**: Combines summarized + recent chat history
* 💬 **Conversational Interface** using Streamlit
* 🤖 **LLM-powered reasoning** with Groq/ChatGPT integration
* 📚 **PDF-to-Chat pipeline** for ingestion from raw brochure files

---

## 🛠️ Tech Stack

| Layer            | Tool/Library                |
| ---------------- | --------------------------- |
| Language Model   | Groq (LLM API)              |
| Vector DB        | Pinecone                    |
| Embeddings       | `BGEEmbedding` custom class |
| Backend          | Python + LangChain          |
| Frontend         | Streamlit                   |
| Document Parsing | Custom OCR pipeline         |

---

## 🗂️ Project Structure

```
scaler-assist/
│
├── src/
│   ├── hist_main_v2.py
│   ├── hist_rag_chain_v2.py
│   ├── hist_retriever.py  
│   ├── embeddings.py     
│   ├── pinecone_utils.py  
│   ├── pdf_extract_and_clean.py
│   ├── text_extract.py 
│   ├── text_cleaner.py   
│   ├── text_splitter.py 
│   ├── create_index.py
│   └── IPYNB/                     
│
├── Context/
│   ├── cleaned_text.txt
│   ├── extracted_text.txt
│   ├── PDFs/
│   └── Texts/
│
└── requirements.txt
```

---

## 🔄 Lifecycle Pipeline

1. **Data Ingestion**:

   * Run `pdf_extract_and_clean.py` to:
     
     * Extract raw text from PDFs using OCR
     * Merge and clean it
     * Store final text in `cleaned_text.txt`

2. **Vectorization**:

   * `text_splitter.py` splits `cleaned_text.txt` into chunks
   * `embeddings.py` embeds them using BGE embeddings
   * `pinecone_utils.py` uploads embeddings to Pinecone

3. **RAG Chain**:

   * `hist_rag_chain_v2.py` builds a hybrid memory chain:
     
     * `ConversationSummaryMemory` (long-term context)
     * `ConversationBufferWindowMemory` (recent chat turns)
       
   * Uses Groq LLM for answering questions and summarizing

4. **Query Interface**:

   * `hist_main_v2.py` sets up a Streamlit chatbot interface
   * Preserves chat state and clears memory on "New Chat"

---

## 🌐 Live Demo

Try the app live on **Streamlit Cloud**:
👉 [**scaler-assist-ai.streamlit.app**](https://scaler-assist-ai.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://scaler-assist-ai.streamlit.app/)

---

## 💬 Example Queries

* "What is the eligibility for Scaler's AI/ML track?"
* "List the business electives in Scaler School of Business."
* "How much hike do students see after the program?"
* "Tell me about Scaler's placement support."
* "What is taught in the intermediate data analytics course?"

---

## 🙌 Acknowledgements

* [Scaler Academy](https://www.scaler.com)
* [LangChain](https://www.langchain.com)
* [Pinecone](https://www.pinecone.io)
* [GROQ](https://www.groq.com)
* [HuggingFace](https://huggingface.co/)

---
