# app/main.py
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings

# Silence Chroma telemetry (recommended for on-prem)
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("code-bot")

# Silence Chroma telemetry log noise
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.chat_models import ChatOpenAI
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

def get_embeddings(provider: str | None = None):
    """
    IMPORTANT: this MUST match how you built the index.
    """
    provider = provider or settings.embedding_provider
    try:
        if provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            base = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001", 
                google_api_key=settings.google_api_key
            )
            return RetryEmbeddings(base, max_retries=5, delay=10)
    except Exception as e:
        print(f"DEBUG: ERROR in get_embeddings: {e}")
        raise e
        
    if provider == "openai":
        return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
    
    # default to local/ollama
    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url
    )


def get_llm(provider: str = "ollama"):
    try:
        if provider == "google":
            if not settings.google_api_key:
                raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set")
            
            return ChatGoogleGenerativeAI(
                model=settings.google_model,
                google_api_key=settings.google_api_key,
                temperature=0.0,
                convert_system_message_to_human=True
            )
    except Exception as e:
        raise e
    if provider == "openai":
         if not settings.openai_api_key:
             raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
         return ChatOpenAI(openai_api_key=settings.openai_api_key, temperature=0.0)
    
    # default to local/ollama
    return ChatOllama(model=getattr(settings, "ollama_model", "llama3.1:8b"))


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def make_chain(k_neighbors: int, provider: str = "ollama"):
    embeddings = get_embeddings(provider)

    # Dynamically select persistent directory based on provider
    if provider == "google":
        persist_dir = settings.chroma_dir_google
    elif provider == "openai":
        persist_dir = settings.chroma_dir_openai
    else:
        persist_dir = settings.chroma_dir_local

    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="local-rag",
    )

    retriever = db.as_retriever(search_kwargs={"k": k_neighbors})

    # Universal Code Intelligence System Prompt
    system_prompt = (
        "You are the 'Universal Code Intelligence Engine', a world-class AI specialized in deep code analysis, "
        "architectural auditing, and cross-project reasoning. "
        "Your mission is to provide precisely tailored insights to any stakeholder based on the provided codebase context."
        "\n\n--- CONTEXT START ---\n{context}\n--- CONTEXT END ---\n\n"
        "### ANALYTICAL PROTOCOL:\n"
        "1. **Audience Detection & Adaption**:\n"
        "   - **Architect**: Provide high-level structural patterns, coupling/cohesion analysis, security posture, and cross-service infrastructure impacts. focus on the 'Why' and the 'Big Picture'.\n"
        "   - **Developer**: Focus on implementation details, API usage, library dependencies, refactoring opportunities, and code logic. Provide snippets and line-by-line explanations.\n"
        "   - **Tester**: Identify REST endpoints, payload structures, validation logic, edge cases, and areas requiring unit or integration tests.\n"
        "   - **Business/BA**: Distill technical complexity into functional value. Explain 'What this does' and 'How it affects the user' without technical jargon.\n"
        "2. **Chain-of-Thought Reasoning**: For complex tasks (e.g., 'Is this secure?'), think step-by-step: \n"
        "   a) Inventory relevant components (Controllers, Filters, Configs).\n"
        "   b) Trace the data flow or logic sequence.\n"
        "   c) Evaluate against industry best practices (OWASP for security, SOLID for design).\n"
        "   d) Formulate the specific conclusion.\n"
        "3. **Synthesized Cross-Referencing**: Never look at files in isolation. Connect Java logic to YAML/XML configs, and relate code to documentation (README/Markdown).\n"
        "4. **Strict Grounding**: Only answer based on the CONTEXT. If the information is missing, describe what is present and exactly what is missing to provide a full answer (e.g., 'I see the auth controller but the bypass logic is not in the provided snippets').\n"
        "5. **Formatting**: Use professional Markdown. Bold key terms. Use code blocks for all technical references. Always cite the source project and file (e.g., `[MyProject] ServiceImpl.java`)."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question:\n{question}\n\nAnswer:"),
        ]
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
async def startup():
    logger.info("Application starting up...")
    try:
        # Pre-warm with default
        provider = settings.embedding_provider
        app.state.chain = make_chain(settings.k_neighbors, provider)
        logger.info(f"Application startup complete for provider: {provider}. Ready for requests.")
    except Exception as e:
        import traceback
        logger.error(f"FATAL ERROR during startup: {e}\n{traceback.format_exc()}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/projects")
async def get_projects():
    """Returns a list of unique project names found in the vector store."""
    try:
        # We need an embedder just to initialize the Chroma object
        # but we won't actually perform a search
        embeddings = get_embeddings(settings.embedding_provider)
        db = Chroma(
            persist_directory=settings.active_chroma_dir,
            embedding_function=embeddings,
            collection_name="local-rag",
        )
        
        # Get all metadatas to extract project names
        # In a very large DB, this might be slow, but for hackathon scale it's fine.
        results = db.get(include=['metadatas'])
        projects = set()
        if results and 'metadatas' in results:
            for meta in results['metadatas']:
                proj = meta.get('project')
                if proj:
                    projects.add(proj)
        
        return {"projects": sorted(list(projects))}
    except Exception as e:
        logger.error(f"Error fetching projects: {e}")
        return {"projects": []}


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        k = req.k or settings.k_neighbors
        provider = req.provider or settings.embedding_provider

        # Rebuild chain to ensure correct provider/k for this request
        chain = make_chain(k, provider)
        result = await chain.ainvoke(req.query)

        sources = []
        for d in result.get("docs", []):
            src = d.metadata.get("source", "unknown")
            if src not in sources:
                sources.append(src)

        return {"answer": result.get("answer", "").strip(), "sources": sources}

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"CHAT ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
