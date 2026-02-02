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


class ConversationRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    use_functions: bool = True


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

    # Dual-Mode Code Intelligence System Prompt
    # Defaults to business-friendly, switches to comprehensive technical mode when needed
    system_prompt = (
        "You are an expert code analyst capable of explaining systems to both business stakeholders and senior developers. "
        "Adapt your response depth based on the user's question.\n\n"
        "--- CONTEXT START ---\n{context}\n--- CONTEXT END ---\n\n"
        
        "**Audience Detection - CRITICAL**\n"
        "Detect the user's expertise level from their question:\n\n"
        
        "**BUSINESS MODE** (Default) - Use when user asks:\n"
        "- 'What is...?', 'How does X work?', 'What happens when...?'\n"
        "- Questions about processes, flows, or business logic\n"
        "- No technical terms like class, method, implementation, code\n"
        "→ Respond in plain language, focus on WHAT and WHY\n"
        "→ Explain as step-by-step business flows\n"
        "→ Skip: testing, error handling, config, code snippets, 'what's missing'\n\n"
        
        "**TECHNICAL MODE** - Switch when user:\n"
        "- Asks 'how is X implemented?', 'show me the code', 'what class/method...?'\n"
        "- Uses technical terms: DTO, API, endpoint, annotation, interface, pattern\n"
        "- Requests 'technical details', 'implementation', 'architecture'\n"
        "- Says 'I'm a developer' or asks about debugging/testing\n"
        "→ Provide comprehensive technical analysis\n"
        "→ Include: class names, method signatures, code patterns, annotations\n"
        "→ Cover: data flows, error handling, async processing, DB operations\n"
        "→ Reference specific files, line-level details, design patterns\n\n"
        
        "**Response Guidelines**\n"
        "1. Start with a direct answer (1-2 sentences)\n"
        "2. Match depth to audience - don't over-explain to experts, don't overwhelm business users\n"
        "3. Use step-by-step flows for processes (step1 → step2 → step3)\n"
        "4. Cite source files at the end\n"
        "5. If information exists in context, FIND IT - search all documents thoroughly\n"
        "6. Look for *Response, *Payload, *DTO patterns for data structures\n\n"
        
        "**What to ALWAYS SKIP (unless explicitly asked)**\n"
        "- 'What's Missing' sections\n"
        "- Testing configurations and test files\n"
        "- Generic error handling explanations\n"
        "- Infrastructure/deployment details\n"
        "- Speculation about code not in context"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", 
             "Question: {question}\n\n"
             "Detect audience from the question and respond appropriately:\n"
             "- Business question → plain language, process flows, skip technical details\n"
             "- Technical question → comprehensive analysis, code details, implementation specifics\n\n"
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


@app.get("/conversation/{conversation_id}/history")
async def get_conversation_history(conversation_id: str):
    """Get conversation history for debugging and context"""
    try:
        from app.conversation_agent import ConversationMemory
        memory = ConversationMemory()
        history = memory.get_conversation_history(conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "turns": len(history),
            "history": [
                {
                    "timestamp": turn.timestamp.isoformat(),
                    "user_message": turn.user_message,
                    "ai_response": turn.ai_response[:200] + "..." if len(turn.ai_response) > 200 else turn.ai_response,
                    "tools_used": turn.tools_used,
                    "sources": turn.sources
                }
                for turn in history
            ]
        }
    except Exception as e:
        return {"error": str(e), "conversation_id": conversation_id}


@app.post("/conversation")
async def conversation_chat(req: ConversationRequest):
    """
    Enhanced conversational interface with memory and function calling
    Uses existing Google Gemini setup to provide agent-like capabilities
    """
    try:
        from app.conversation_agent import create_conversation_agent
        
        agent = await create_conversation_agent()
        response = await agent.chat(
            query=req.query,
            conversation_id=req.conversation_id
        )
        
        return {
            "answer": response["answer"],
            "conversation_id": response["conversation_id"], 
            "tools_used": response["tools_used"],
            "sources": response["sources"],
            "reasoning": response.get("reasoning"),
            "function_results": response.get("function_results")
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Conversation Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        # Fallback to standard chat
        fallback_response = await chat(ChatRequest(query=req.query))
        return {
            **fallback_response,
            "conversation_id": req.conversation_id,
            "fallback_used": True
        }


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
