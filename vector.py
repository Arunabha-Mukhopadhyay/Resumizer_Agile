# vector.py
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class SimpleDocstore:
    """Fallback docstore used when embedding backends are unavailable."""

    def __init__(self, doc: Document):
        self._dict = {"0": doc}


class SimpleVectorStore:
    """Mimics the `docstore` attribute structure expected downstream."""

    def __init__(self, doc: Document):
        self.docstore = SimpleDocstore(doc)

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
        print("[ℹ️] Falling back to simple in-memory store (no embeddings).")
        return SimpleVectorStore(doc)
