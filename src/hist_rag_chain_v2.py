from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from typing import List, Dict, Any

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

class HybridMemory:
    """
    Hybrid memory that combines:
    - ConversationSummaryMemory for long-term summarized history
    - ConversationBufferWindowMemory for recent raw conversation turns
    """
    
    def __init__(self, llm, window_size: int = 4, summary_memory_key: str = "summary_history", window_memory_key: str = "recent_history"):
        """
        Initialize hybrid memory.
        
        Args:
            llm: Language model for summarization
            window_size: Number of recent conversation turns to keep in raw format
            summary_memory_key: Key for summary memory
            window_memory_key: Key for window memory
        """
        self.summary_memory = ConversationSummaryMemory(
            llm=llm,
            memory_key=summary_memory_key,
            return_messages=True
        )
        
        self.window_memory = ConversationBufferWindowMemory(
            k=window_size,
            memory_key=window_memory_key,
            return_messages=True
        )
        
        self.summary_memory_key = summary_memory_key
        self.window_memory_key = window_memory_key
        
    def add_user_message(self, message: str):
        """Add user message to both memories"""
        self.summary_memory.chat_memory.add_user_message(message)
        self.window_memory.chat_memory.add_user_message(message)
        
    def add_ai_message(self, message: str):
        """Add AI message to both memories"""
        self.summary_memory.chat_memory.add_ai_message(message)
        self.window_memory.chat_memory.add_ai_message(message)
        
    def get_combined_history(self) -> List[BaseMessage]:
        """
        Get combined chat history with summary + recent raw messages.
        Returns summarized history followed by recent raw conversation turns.
        """
        messages = []
        
        # Get summary if available
        summary_vars = self.summary_memory.load_memory_variables({})
        if summary_vars.get(self.summary_memory_key):
            summary_messages = summary_vars[self.summary_memory_key]
            if summary_messages and len(summary_messages) > 0:
                # Add summary as a system-like message for context
                if isinstance(summary_messages, list):
                    messages.extend(summary_messages)
                else:
                    messages.append(AIMessage(content=f"Previous conversation summary: {summary_messages}"))
        
        # Get recent raw messages from window memory
        window_vars = self.window_memory.load_memory_variables({})
        recent_messages = window_vars.get(self.window_memory_key, [])
        
        # Only include recent messages that aren't already in summary
        # This prevents duplication of the most recent messages
        if recent_messages:
            messages.extend(recent_messages)
            
        return messages
    
    def clear(self):
        """Clear both memories"""
        self.summary_memory.clear()
        self.window_memory.clear()

def create_rag_chain(retriever, groq_api_key, model, memory=None, window_size: int = 4):
    """
    Create RAG chain with hybrid memory approach.
    
    Args:
        retriever: Document retriever
        groq_api_key: Groq API key
        model: Model name
        memory: Optional existing memory (will be replaced with hybrid memory)
        window_size: Number of recent turns to keep in raw format
    """
    llm = ChatGroq(groq_api_key=groq_api_key, model=model)
    
    # Create hybrid memory
    hybrid_memory = HybridMemory(llm=llm, window_size=window_size)
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Create question answering chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create full RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Wrapper class to maintain compatibility with existing code
    class HybridRAGChainWrapper:
        def __init__(self, chain, hybrid_memory):
            self.chain = chain
            self.hybrid_memory = hybrid_memory
            
        def __call__(self, inputs):
            # Get combined chat history from hybrid memory
            chat_history = self.hybrid_memory.get_combined_history()
            
            # Run the chain
            result = self.chain.invoke({
                "input": inputs.get("question", inputs.get("query", "")),
                "chat_history": chat_history
            })
            
            # Update hybrid memory
            question = inputs.get("question", inputs.get("query", ""))
            answer = result["answer"]
            
            self.hybrid_memory.add_user_message(question)
            self.hybrid_memory.add_ai_message(answer)
            
            # Return in expected format
            return {
                "answer": answer,
                "source_documents": result.get("context", [])
            }
            
        def invoke(self, inputs):
            return self(inputs)
        
        def clear_memory(self):
            """Clear conversation memory"""
            self.hybrid_memory.clear()
        
        def get_memory_stats(self) -> Dict[str, Any]:
            """Get statistics about memory usage"""
            summary_vars = self.hybrid_memory.summary_memory.load_memory_variables({})
            window_vars = self.hybrid_memory.window_memory.load_memory_variables({})
            
            return {
                "summary_length": len(str(summary_vars.get(self.hybrid_memory.summary_memory_key, ""))),
                "recent_messages_count": len(window_vars.get(self.hybrid_memory.window_memory_key, [])),
                "total_messages_in_summary": len(self.hybrid_memory.summary_memory.chat_memory.messages),
                "window_size": self.hybrid_memory.window_memory.k
            }
    
    return HybridRAGChainWrapper(rag_chain, hybrid_memory)