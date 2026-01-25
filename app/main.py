# app/main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings

# Silence Chroma telemetry (recommended for on-prem)
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI(title="Local RAG (OpenAI / Ollama + Chroma)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    k: int | None = None
    provider: str | None = "ollama"  # "ollama", "openai", "google"


class RetryEmbeddings:
    """Wrapper to retry embeddings on rate limit errors."""
    def __init__(self, base_embeddings, max_retries=3, delay=5):
        self.base_embeddings = base_embeddings
        self.max_retries = max_retries
        self.delay = delay

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        import time
        errors = []
        for i in range(self.max_retries):
            try:
                return self.base_embeddings.embed_query(text)
            except Exception as e:
                errors.append(e)
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"Rate limit hit, retrying in {self.delay}s...")
                    time.sleep(self.delay)
                    # Exponential backoff
                    self.delay *= 2
                else:
                    raise e
        raise errors[-1]

def get_embeddings():
    """
    IMPORTANT: this MUST match how you built the index.
    """
    provider = settings.embedding_provider
    try:
        if provider == "google":
            print(f"DEBUG: Initializing google embeddings...")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            base = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=settings.google_api_key,
                # api_version="v1" 
            )
            print(f"DEBUG: Google embeddings initialized.")
            return RetryEmbeddings(base, max_retries=5, delay=10)
    except Exception as e:
        print(f"DEBUG: ERROR in get_embeddings: {e}")
        raise e
        
    if provider == "openai":
        return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
    
    # default to local/ollama
    return HuggingFaceEmbeddings(model_name=settings.local_embedding_model)


def get_llm(provider: str = "ollama"):
    try:
        if provider == "google":
            if not settings.google_api_key:
                raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set")
            
            return ChatGoogleGenerativeAI(
                model=settings.google_model,
                google_api_key=settings.google_api_key,
                temperature=0.1,
                convert_system_message_to_human=True
            )
    except Exception as e:
        raise e
    if provider == "openai":
         if not settings.openai_api_key:
             raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
         return ChatOpenAI(openai_api_key=settings.openai_api_key, temperature=0.1)
    
    # default to local/ollama
    return ChatOllama(model=getattr(settings, "ollama_model", "llama3.1:8b"))


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def make_chain(k_neighbors: int, provider: str = "ollama"):
    embeddings = get_embeddings()

    db = Chroma(
        persist_directory=settings.active_chroma_dir,
        embedding_function=embeddings,
        collection_name="local-rag",
    )

    retriever = db.as_retriever(search_kwargs={"k": k_neighbors})

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use ONLY the context below.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )

    llm = get_llm(provider)

    base = RunnableParallel(
        docs=retriever,
        question=RunnablePassthrough(),
    )

    to_prompt_inputs = RunnableLambda(
        lambda x: {
            "context": format_docs(x["docs"]),
            "question": x["question"],
            "docs": x["docs"],
        }
    )

    answer_chain = prompt | llm | StrOutputParser()

    final = RunnableParallel(
        answer=RunnableLambda(
            lambda x: {"context": x["context"], "question": x["question"]}
        )
        | answer_chain,
        docs=RunnableLambda(lambda x: x["docs"]),
    )

    return base | to_prompt_inputs | final


@app.on_event("startup")
def startup():
    # Pre-warm with default
    app.state.chain = make_chain(settings.k_neighbors, "ollama")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        k = req.k or settings.k_neighbors

        k = req.k or settings.k_neighbors
        provider = req.provider or "ollama"

        # If params match the cached chain's configuration (simplified check), use it.
        # But here we have dynamic providers, so we might just rebuild the chain or cache by provider.
        # For simplicity, let's just rebuild if it's not the default provider or if k changed.
        # Ideally, we should cache chains by (k, provider).
        
        # Simple Logic: Always make chain for now to ensure provider switch works instantly 
        # (Chroma retriever initialization is light usually, but optimal would be caching).
        chain = make_chain(k, provider)
        result = await chain.ainvoke(req.query)

        sources = []
        for d in result.get("docs", []):
            src = d.metadata.get("source", "unknown")
            if src not in sources:
                sources.append(src)

        return {"answer": result.get("answer", "").strip(), "sources": sources}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
