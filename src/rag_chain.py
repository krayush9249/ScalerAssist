from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

def format_inputs(inputs):
    return {
        "context": "\n\n".join([doc.page_content for doc in inputs["documents"]]),
        "question": inputs["question"]
    }

def create_rag_chain(retriever_runnable, groq_api_key, model):
    llm = ChatGroq(groq_api_key=groq_api_key, model=model)
    output_parser = StrOutputParser()

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an intelligent assistant for Scaler Academy, trained on internal documents, placement records, "
         "program curricula, and student feedback. "
         "Answer questions only based on the context provided. "
         "If the answer is not found in the context, reply with: "
         "“I'm sorry, I couldn't find that information in the available documents.” "
         "Be precise, concise, and maintain a professional and helpful tone."),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    rag_chain = (
        RunnableMap({
            "documents": retriever_runnable,
            "question": lambda x: x["question"]
        })
        | format_inputs
        | chat_prompt
        | llm
        | output_parser
    )
    return rag_chain

