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
                temperature=0.1,
                max_output_tokens=2048,
                convert_system_message_to_human=True
            )
    except Exception as e:
        raise e
    if provider == "openai":
         if not settings.openai_api_key:
             raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
         return ChatOpenAI(openai_api_key=settings.openai_api_key, temperature=0.1, max_tokens=2048)
    
    # default to local/ollama
    return ChatOllama(
        model=getattr(settings, "ollama_model", "llama3.1:8b"),
        temperature=0.1,
        num_ctx=4096
    )


def format_docs(docs):
    """Enhanced document formatting with metadata context for better understanding."""
    formatted_sections = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get('source', 'unknown')
        project = d.metadata.get('project', 'unknown')
        filename = d.metadata.get('filename', 'unknown')
        
        section = f"=== DOCUMENT {i}: [{project}] {filename} ===\n"
        section += f"Source: {source}\n\n"
        section += d.page_content
        formatted_sections.append(section)
    
    return "\n\n".join(formatted_sections)


def make_chain(k_neighbors: int, provider: str = "ollama"):
    embeddings = get_embeddings(provider)

    # Dynamically select persistent directory based on provider
    if provider == "google":
        persist_dir = settings.resolved_chroma_dir_google
    elif provider == "openai":
        persist_dir = settings.resolved_chroma_dir_openai
    else:
        persist_dir = settings.resolved_chroma_dir_local

    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="local-rag",
    )

    retriever = db.as_retriever(search_kwargs={"k": k_neighbors})

    # Universal Code Intelligence System Prompt - Comprehensive Analysis Mode
    system_prompt = (
        "You are a comprehensive code analyst that provides detailed, thorough answers from the RAG context. Always anticipate what technical details users need.\n\n"
        "--- CONTEXT START ---\n{context}\n--- CONTEXT END ---\n\n"
        "**Mission**\n"
        "- Provide comprehensive, detailed answers grounded ONLY in the context\n"
        "- Anticipate follow-up questions and answer them proactively in your initial response\n"
        "- Assume users may not know what technical details to ask for - provide them anyway\n\n"
        "**Core Behaviors - BE COMPREHENSIVE FIRST**\n"
        "- Always provide full technical details available in the context upfront\n"
        "- Include specific API endpoints, class names, method signatures, and technical flows\n"
        "- Explain the 'how' and 'why' with step-by-step processes when describing system interactions\n"
        "- Cover multiple aspects: data flow, error handling, background processes, database operations\n"
        "- State conflicts neutrally when sources disagree, but include all available information\n\n"
        "**Detailed Analysis Requirements - PRIORITIZE PROCESS FLOWS**\n"
        "1) **Process Sequences/Pipelines**: ALWAYS look for and highlight step-by-step business flows (e.g., step1->step2->step3, method chains, sequential operations)\n"
        "2) **API Flows**: Always include specific endpoints, HTTP methods, request/response formats, error handling\n"
        "3) **System Interactions**: Detail which services call which others, with class/method names and purposes\n"
        "4) **Data Processing**: Explain validation steps, transformations, persistence mechanisms, async operations\n"
        "5) **Background Processes**: Describe schedulers, queues, triggers, timing, AND the actual processing logic they execute\n"
        "6) **Technical Implementation**: Include configuration properties, annotations, design patterns\n\n"
        "**Process Flow Detection - CRITICAL**\n"
        "- Search for method names that suggest sequential steps (e.g., determine*, fetch*, perform*, submit*, update*)\n"
        "- Look for workflow patterns, pipeline implementations, state machines\n"
        "- Identify business logic sequences even when wrapped in infrastructure code\n"
        "- Always distinguish between 'how the process is triggered' vs 'what the process actually does'\n"
        "- When explaining schedulers/queues, ALWAYS include what business logic they execute\n"
        "**Proactive Detail Provision**\n"
        "- Code snippets and class names when explaining functionality\n"
        "- Configuration values and their purposes\n"
        "- Database operations and persistence strategies  \n"
        "- Error handling and exception management\n"
        "- Async/background processing details\n"
        "- Integration patterns and adapter implementations\n\n"
        "**Comprehensive Formatting - PROCESS FLOW FIRST**\n"
        "- **Business Process Flow** — ALWAYS start with step-by-step sequences when they exist (e.g., step1 → step2 → step3)\n"
        "- **Detailed Answer** — comprehensive explanation with technical specifics\n"
        "- **Implementation Details** — class names, method signatures, specific technical flows\n"
        "- **Infrastructure Context** — scheduling, queuing, error handling (but never replace process flows)\n"
        "- **Technical Specifications** — API endpoints, configuration, database operations\n"
        "- **Sources** — specific file paths and relevant code sections\n"
        "- **What's Missing** — only when critical information is genuinely unavailable\n\n"
        "**Evidence and Citations**\n"
        "- Always cite specific files, classes, and methods\n"
        "- Include configuration properties and their values when relevant\n"
        "- Reference specific code patterns and architectural decisions\n"
        "- Quote relevant code snippets when they clarify the explanation\n\n"
        "**Style - COMPREHENSIVE & ACCESSIBLE**\n"
        "- Write for mixed audiences: technical developers AND non-technical stakeholders\n"
        "- Define technical terms when first introduced (e.g., 'DTO (Data Transfer Object)')\n"
        "- Use analogies or business context when explaining complex technical flows\n"
        "- Structure answers with clear sections and bullet points for readability\n"
        "- Include 'business impact' or 'why this matters' context when relevant\n"
        "- Provide operational details: timing, scheduling, resource usage, performance implications\n"
        "- Always explain the relationship between components (how they work together)"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", 
             "Question: {question}\n\n"
             "Please provide a comprehensive answer that includes:\n"
             "• **BUSINESS PROCESS FLOWS FIRST**: Step-by-step sequences (e.g., determineX → fetchY → performZ)\n"
             "• Complete technical implementation details\n"
             "• Specific class names, methods, and API endpoints\n"
             "• Background processes AND the business logic they execute\n"
             "• Database operations and data persistence\n"
             "• Error handling and exception management\n"
             "• Configuration settings and their purposes\n\n"
             "CRITICAL: When explaining schedulers/processors/queues, always include both:\n"
             "1) HOW the process is triggered (scheduling/infrastructure)\n"
             "2) WHAT the process actually does (business logic steps)\n\n"
             "Answer:"
            ),
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

        # Use cached chain if parameters match defaults
        if k == settings.k_neighbors and provider == settings.embedding_provider:
            chain = app.state.chain  # Use existing cache
        else:
            chain = make_chain(k, provider)  # Only rebuild when needed
        
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
