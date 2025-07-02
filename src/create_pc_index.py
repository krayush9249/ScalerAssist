import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

env_path = "/Users/kumarpersonal/Downloads/ScalerAssist/venv-scaler-assist/.env"
load_dotenv(dotenv_path=env_path)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=pinecone_api_key)

if pinecone_index_name not in pc.list_indexes().names():
    print(f"Creating index {pinecone_index_name}...")

    try:
        pc.create_index(
            name=pinecone_index_name,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

        while True:
            status = pc.describe_index(pinecone_index_name).status
            if status.get("ready"):
                print(f"Index {pinecone_index_name} is ready.")
                break
            else:
                print(f"Index {pinecone_index_name} is still being created. Waiting...")
                time.sleep(5)

    except Exception as e:
        print(f"Index creation failed: {e}")
else:
    print(f"Index {pinecone_index_name} already exists.")
