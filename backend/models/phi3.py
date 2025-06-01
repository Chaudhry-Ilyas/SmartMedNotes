import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError(
        "llama_cpp not found. Please install it with: "
        "pip install llama-cpp-python"
    )

# Configuration
MODEL_PATH = "./phi3-finetuned.q5_k_m.gguf"  # Download this file first
FAISS_INDEX_PATH = "./processed_data/combined_faiss_index.faiss"
DOC_DATA_PATH = "./processed_data/document_data.json"

# Verify files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Please download the phi-3-mini GGUF model first."
    )

# Load FAISS index and document data
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(DOC_DATA_PATH, "r") as f:
    document_data = json.load(f)

print(f"✅ FAISS Index: {faiss_index.ntotal} vectors")
print(f"✅ Documents: {len(document_data)} entries")

# Load embedding model
retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize LLM with error handling
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=4,
        n_gpu_layers=0  # Set to >0 if you have GPU
    )
except Exception as e:
    raise RuntimeError(f"Failed to load LLM model: {str(e)}")

def retrieve_faiss_docs(query, top_k=3):
    query_emb = retriever.encode(query, convert_to_numpy=True).astype(np.float32).reshape(1, -1)
    distances, indices = faiss_index.search(query_emb, top_k)
    
    return [document_data[i]["content"] for i in indices[0] if i < len(document_data)]

def format_prompt(context, query):
    return f"""<|user|>
You are a helpful orthopedic AI assistant. Answer the question based on the given context.

Context: {context}

Question: {query}
<|assistant|>
"""

def generate_response(query):
    try:
        docs = retrieve_faiss_docs(query)
        if not docs:
            return "🔍 No relevant medical information found"
        
        context = "\n".join(docs[:3])  # Use top 3 docs
        prompt = format_prompt(context, query)
        
        output = llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )
        
        return output['choices'][0]['text'].strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
