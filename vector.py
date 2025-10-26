# vector.py
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_local_model():
    print("[🧠] Using HuggingFace Embeddings (local)")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def generate_vector_store(text: str):
    doc = Document(page_content=text)
    try:
        embeddings = get_local_model()
        return FAISS.from_documents([doc], embeddings)
    except Exception as e:
        print(f"[❌] Failed to generate local vector store: {e}")
        raise RuntimeError("Failed to generate vector store with local embeddings.")