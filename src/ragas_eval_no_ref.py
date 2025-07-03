# RAGAS Evaluation Without Reference Answers

import os
import sys
from dotenv import load_dotenv
from hist_retriever import CustomPineconeRetriever
from hist_rag_chain_v2 import create_rag_chain
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_groq import ChatGroq

# Load environment variables
env_path = "/Users/kumarpersonal/Downloads/ScalerAssist/venv-scaler-assist/.env"
load_dotenv(dotenv_path=env_path)

# Set up environment
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'ScalerAssist'
os.environ["OPENAI_API_KEY"] = "sk-placeholder"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INFER_MODEL_NAME = os.getenv("INFER_MODEL_NAME")

def run_ragas_eval(query: str, answer: str, retrieved_docs):
    """
    Run RAGAS evaluation on the given query, generated answer, and retrieved documents.
    """
    contexts_text = [doc.page_content for doc in retrieved_docs]
    ragas_data = Dataset.from_dict({
        "question": [query],
        "answer": [answer],
        "contexts": [contexts_text],
    })

    # Create Groq LLM instance for RAGAS
    groq_llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=INFER_MODEL_NAME,
        temperature=0
    )

    results = evaluate(
        ragas_data,
        metrics=[faithfulness, answer_relevancy],
        llm=groq_llm
    )
    
    return results

def run_evaluation_batch(test_questions):
    """
    Run evaluation on a batch of test questions
    """
    # Initialize retriever and RAG chain
    retriever = CustomPineconeRetriever()
    rag_chain = create_rag_chain(
        retriever=retriever,
        groq_api_key=GROQ_API_KEY,
        model=INFER_MODEL_NAME,
        window_size=4
    )
    
    results = []
    
    for i, question in enumerate(test_questions):
        print(f"\nEvaluating question {i+1}/{len(test_questions)}: {question}")
        
        try:
            # Get answer from RAG chain
            result = rag_chain.invoke({
                "question": question,
                "chat_history": []
            })
            
            answer = result["answer"]
            retrieved_docs = result.get("source_documents", [])
            
            print(f"Answer: {answer[:100]}...")
            
            # Run evaluations
            if retrieved_docs:
                ragas_results = run_ragas_eval(question, answer, retrieved_docs)
                
                evaluation_result = {
                    "question": question,
                    "answer": answer,
                    "ragas_metrics": ragas_results,
                }
                
                results.append(evaluation_result)
                print(f"RAGAS scores: {ragas_results}")
            else:
                print("No retrieved documents found")
                
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            
    return results

def main():
    # Define test questions
    test_questions = [
        "What is Scaler Academy?",
        "What courses does Scaler offer?",
        "How long is the Data Science program?",
        "What are the admission requirements?",
        "What is the fee structure?"
    ]
    
    if len(sys.argv) > 1:
        # Allow single question from command line
        test_questions = [sys.argv[1]]
    
    print("Starting evaluation...")
    results = run_evaluation_batch(test_questions)
    
    # Save results to file
    # import json
    # with open("evaluation_results.json", "w") as f:
    #     json.dump(results, f, indent=2)
    
    # print(f"\nEvaluation complete! Results saved to evaluation_results.json")

if __name__ == "__main__":
    main()