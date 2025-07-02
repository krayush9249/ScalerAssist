from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory

# Contextualize question prompt for history-aware retrieval
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Main QA prompt
qa_system_prompt = (
    "You are an intelligent assistant for Scaler Academy. "
    "You have access to internal documents, placement records, curricula, and student feedback.\n\n"
    "Maintain a professional tone. Answer only from the context.\n\n"
    "If the answer isn't available, say:\n"
    "\"I'm sorry, I couldn't find that information.\"\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def create_rag_chain(retriever, groq_api_key, model, memory=None):
    llm = ChatGroq(groq_api_key=groq_api_key, model=model)
    
    # Memory fallback
    if memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Create question answering chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create full RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Wrapper class to maintain compatibility with existing code
    class RAGChainWrapper:
        def __init__(self, chain, memory):
            self.chain = chain
            self.memory = memory
            
        def __call__(self, inputs):
            # Get chat history from memory
            chat_history = self.memory.chat_memory.messages
            
            # Run the chain
            result = self.chain.invoke({
                "input": inputs.get("question", inputs.get("query", "")),
                "chat_history": chat_history
            })
            
            # Update memory
            question = inputs.get("question", inputs.get("query", ""))
            answer = result["answer"]
            
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
            
            # Return in expected format
            return {
                "answer": answer,
                "source_documents": result.get("context", [])
            }
            
        def invoke(self, inputs):
            return self(inputs)
    
    return RAGChainWrapper(rag_chain, memory)