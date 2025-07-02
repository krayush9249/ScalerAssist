# import os
# from dotenv import load_dotenv
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
# from datasets import Dataset
# from langchain_community.llms import HuggingFaceHub
# from langchain.evaluation import CriteriaEvalChain
# from langsmith import Client
# from langsmith import RunTree

# # Load environment variables
# env_path = "/Users/kumarpersonal/Downloads/ScalerAssist/venv-scaler-assist/.env"
# load_dotenv(dotenv_path=env_path)

# HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN")
# LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# client = Client(api_key=LANGCHAIN_API_KEY)  # Initialize Langsmith client 

# def run_ragas_eval(query: str, answer: str, retrieved_docs):
#     """
#     Run RAGAS evaluation on the given query, generated answer, and retrieved documents,
#     and log the run to Langsmith.
#     """
#     contexts_text = [doc.page_content for doc in retrieved_docs]
#     ragas_data = Dataset.from_dict({
#         "question": [query],
#         "answer": [answer],
#         "contexts": [contexts_text],
#     })

#     with client.run_manager(
#         input={"query": query, "answer": answer, "contexts": contexts_text},
#         run_type=RunTypeEnum.tool,
#         tags=["ragas-eval"]
#     ) as run:
#         results = evaluate(
#             ragas_data,
#             metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
#         )
#         run.metadata = results  # Save evaluation results as run metadata
#     return results

# def run_langchain_criteria_eval(query: str, answer: str, criteria: str = "Correctness, Helpfulness, Relevance, Faithfulness"):
#     """
#     Run LangChain's CriteriaEvalChain to evaluate the generated answer against specified criteria,
#     and log the run to Langsmith.
#     """
#     llm = HuggingFaceHub(
#         repo_id="google/flan-t5-base",
#         model_kwargs={"temperature": 0},
#         huggingfacehub_api_token=HF_HUB_TOKEN
#     )
#     evaluator = CriteriaEvalChain.from_llm(llm)

#     with client.run_manager(
#         input={"query": query, "answer": answer, "criteria": criteria},
#         run_type=RunTypeEnum.tool,
#         tags=["criteria-eval"]
#     ) as run:
#         results = evaluator.evaluate(
#             prediction=answer,
#             input=query,
#             criteria=criteria
#         )
#         run.metadata = results  # Save evaluation results as run metadata
#     return results

import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy  # Only metrics that don't require ground truth
from datasets import Dataset
from langchain_huggingface import HuggingFaceEndpoint  # Updated import
from langchain.evaluation import CriteriaEvalChain
from langsmith import Client
from langsmith import RunTree  # Use RunTree for tracking runs

# RAGAS configuration imports
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
env_path = "/Users/kumarpersonal/Downloads/ScalerAssist/venv-scaler-assist/.env"
load_dotenv(dotenv_path=env_path)

HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

client = Client(api_key=LANGCHAIN_API_KEY)  # Initialize Langsmith client

# Configure RAGAS to use Hugging Face instead of OpenAI
def setup_ragas_with_huggingface():
    """Configure RAGAS to use Hugging Face models instead of OpenAI"""
    try:
        # Create Hugging Face LLM
        hf_llm = HuggingFaceEndpoint(
            repo_id="microsoft/DialoGPT-medium",  # Alternative model that works well for evaluation
            temperature=0.1,
            huggingfacehub_api_token=HF_HUB_TOKEN,
            max_new_tokens=512
        )
        
        # Create Hugging Face embeddings
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Wrap for RAGAS
        ragas_llm = LangchainLLMWrapper(hf_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
        
        # Configure metrics with custom LLM and embeddings
        faithfulness.llm = ragas_llm
        faithfulness.embeddings = ragas_embeddings
        answer_relevancy.llm = ragas_llm
        answer_relevancy.embeddings = ragas_embeddings
        
        return ragas_llm, ragas_embeddings
        
    except Exception as e:
        print(f"Error setting up RAGAS with Hugging Face: {e}")
        print("Falling back to simple evaluation without RAGAS LLM configuration")
        return None, None 

def run_ragas_eval(query: str, answer: str, retrieved_docs):
    """
    Run RAGAS evaluation on the given query, generated answer, and retrieved documents,
    and log the run to Langsmith.
    
    Uses only metrics that don't require ground truth:
    - faithfulness: Measures factual accuracy based on retrieved contexts
    - answer_relevancy: Measures how relevant the answer is to the question
    
    Args:
        query: The input question
        answer: The generated answer
        retrieved_docs: List of retrieved documents
    """
    try:
        # Setup RAGAS with Hugging Face
        ragas_llm, ragas_embeddings = setup_ragas_with_huggingface()
        
        contexts_text = [doc.page_content for doc in retrieved_docs]
        
        ragas_data = Dataset.from_dict({
            "question": [query],
            "answer": [answer],
            "contexts": [contexts_text],
        })
        
        # Use only metrics that don't require ground truth
        metrics = [faithfulness, answer_relevancy]

        # Use RunTree for better tracking
        run_tree = RunTree(
            name="ragas-evaluation",
            run_type="tool",
            inputs={"query": query, "answer": answer, "contexts": contexts_text},
            tags=["ragas-eval", "no-ground-truth", "huggingface"]
        )
        
        try:
            results = evaluate(
                ragas_data,
                metrics=metrics
            )
            
            # End the run with results
            run_tree.end(outputs=results)
            run_tree.post()  # Post the run to LangSmith
            
        except Exception as e:
            # End the run with error
            run_tree.end(error=str(e))
            run_tree.post()
            raise e
            
        return results
        
    except Exception as e:
        print(f"Error in RAGAS evaluation: {e}")
        return None

def run_langchain_criteria_eval(query: str, answer: str, criteria: str = "helpfulness"):
    """
    Run LangChain's CriteriaEvalChain to evaluate the generated answer against specified criteria,
    and log the run to Langsmith.
    """
    try:
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-base",
            temperature=0,
            huggingfacehub_api_token=HF_HUB_TOKEN
        )
        evaluator = CriteriaEvalChain.from_llm(llm=llm, criteria=criteria)

        # Use RunTree for better tracking
        run_tree = RunTree(
            name="criteria-evaluation",
            run_type="tool",
            inputs={"query": query, "answer": answer, "criteria": criteria},
            tags=["criteria-eval"]
        )
        
        try:
            results = evaluator.evaluate_strings(
                prediction=answer,
                input=query
                # Removed reference parameter as it's not expected
            )
            
            # End the run with results
            run_tree.end(outputs=results)
            run_tree.post()  # Post the run to LangSmith
            
        except Exception as e:
            # End the run with error
            run_tree.end(error=str(e))
            run_tree.post()
            raise e
            
        return results
        
    except Exception as e:
        print(f"Error in LangChain criteria evaluation: {e}")
        return None

