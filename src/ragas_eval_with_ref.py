# RAGAS Evaluation With Reference Answers

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

def create_reference_dataset():
    """
    Create a dataset with reference answers for complete evaluation
    """
    test_qa_pairs = [
        {
            "question": "What is Scaler Academy?",
            "reference_answer": "Scaler Academy is an online transformative upskilling platform for working tech professionals that offers comprehensive courses in various technology domains."
        },
        {
            "question": "What courses does Scaler offer?",
            "reference_answer": "Scaler offers courses in Advanced AI and Machine Learning, Data Science, Software Engineering, and other technology-focused programs."
        },
        {
            "question": "How long is the Data Science program?",
            "reference_answer": "The Data Science program at Scaler Academy has varying durations based on learner profiles, typically ranging from 1 year to longer depending on the specific track."
        },
        {
            "question": "What are the admission requirements?",
            "reference_answer": "To get admission to Scaler Academy, candidates need to take the Scaler Entrance Test, which consists of 16 questions covering various topics."
        },
        {
            "question": "What is the fee structure?",
            "reference_answer": "The fee structure includes admission fees of ₹1,00,000 (non-refundable) plus additional program fees that vary by course."
        }
    ]
    return test_qa_pairs

def run_evaluation_with_references(test_qa_pairs):
    """
    Run evaluation when you have reference answers
    test_qa_pairs: list of dicts with 'question', 'reference_answer' keys
    """
    retriever = CustomPineconeRetriever()
    rag_chain = create_rag_chain(
        retriever=retriever,
        groq_api_key=GROQ_API_KEY,
        model=INFER_MODEL_NAME,
        window_size=4
    )
    
    results = []
    
    for i, qa_pair in enumerate(test_qa_pairs):
        question = qa_pair["question"]
        reference = qa_pair["reference_answer"]
        
        print(f"\nEvaluating question {i+1}/{len(test_qa_pairs)}: {question}")
        
        try:
            result = rag_chain.invoke({
                "question": question,
                "chat_history": []
            })
            
            answer = result["answer"]
            retrieved_docs = result.get("source_documents", [])
            
            if retrieved_docs:
                contexts_text = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in retrieved_docs]
                
                ragas_data = Dataset.from_dict({
                    "question": [question],
                    "answer": [answer],
                    "contexts": [contexts_text],
                    "reference": [reference]  
                })
                
                groq_llm = ChatGroq(
                    api_key=GROQ_API_KEY,
                    model_name=INFER_MODEL_NAME,
                    temperature=0
                )
                ragas_results = evaluate(
                    ragas_data,
                    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
                    llm=groq_llm
                )
                
                results.append({
                    "question": question,
                    "answer": answer,
                    "reference": reference,
                    "ragas_metrics": ragas_results,
                })
                
                print(f"RAGAS scores: {ragas_results}")
                
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            
    return results

def main():
    print("Starting evaluation...")

    test_qa_pairs = create_reference_dataset()
    
    results = run_evaluation_with_references(test_qa_pairs)
    
    # Optional: save to file
    # import json
    # with open("evaluation_with_refs.json", "w") as f:
    #     json.dump(results, f, indent=2)                

if __name__ == "__main__":
    main()