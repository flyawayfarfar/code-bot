# app/build_index.py
from app.rag_pipeline import build_vector_store

if __name__ == "__main__":
    print("Building vector store from data/ folder...")
    build_vector_store()
    print("Vector store built successfully. Ready for Q&A!")
