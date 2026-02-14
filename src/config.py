import os

from dotenv import load_dotenv

load_dotenv()

# Hugging Face
HF_API_KEY = os.environ.get("HF_API_KEY")
HF_EMBED_MODEL_ID = os.environ.get("HF_EMBED_MODEL_ID", "BAAI/bge-small-en-v1.5")
HF_LLM_MODEL_ID = os.environ.get("HF_LLM_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")

# Milvus
MILVUS_URI = os.environ.get("MILVUS_URI", "./milvus_store/milvus.db")

# Text splitting
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
