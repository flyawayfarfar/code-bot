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


def get_embeddings():
    """
    IMPORTANT: this MUST match how you built the index.
    """
    if settings.use_local_embeddings:
        return HuggingFaceEmbeddings(model_name=settings.local_embedding_model)
    return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)


def get_llm():
    if settings.use_local_llm:
        # NOTE: some versions don't accept temperature kwarg; keep it simple.
        return ChatOllama(model=getattr(settings, "ollama_model", "llama3"))
    return ChatOpenAI(openai_api_key=settings.openai_api_key, temperature=0.1)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def make_chain(k_neighbors: int):
    embeddings = get_embeddings()

    db = Chroma(
        persist_directory=settings.chroma_dir,
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

    llm = get_llm()

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
    app.state.chain = make_chain(settings.k_neighbors)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        k = req.k or settings.k_neighbors

        chain = app.state.chain if k == settings.k_neighbors else make_chain(k)
        result = await chain.ainvoke(req.query)

        sources = []
        for d in result.get("docs", []):
            src = d.metadata.get("source", "unknown")
            if src not in sources:
                sources.append(src)

        return {"answer": result.get("answer", "").strip(), "sources": sources}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
