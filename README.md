# Universal Code Intelligence Engine

![Banner](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.13.2-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.111.0-red)

## What is this?

**Universal Code Intelligence Engine** is a specialized RAG-powered system for deep source code analysis and architectural understanding. It enables developers, architects, and business stakeholders to chat with their codebases using natural language, providing contextual insights about code flows, reusable components, business logic, and security patterns.

The system is designed specifically for enterprise codebases, with intelligent parsing of Java, Python, XML, YAML, and configuration files.

### Key Capabilities

- **Natural Language Code Search**: Ask questions about your codebase in plain English and get relevant code snippets and explanations.
- **Cross-Project Understanding**: Index multiple related projects to find connections and dependencies between them.
- **Multi-Provider Support**: Seamlessly switch between **Ollama** (fully local), **OpenAI**, and **Google AI (Gemini)** for both embedding and chat.
- **Language-Aware Parsing**: Intelligently handles Java, Python, XML, and YAML to improve retrieval quality.
- **Local-First Design**: Built to run entirely on your infrastructure for maximum security and privacy, with local vector storage via ChromaDB.

## Architecture Overview

The system can operate in two modes: a standard RAG pipeline for direct answers (Basic Mode) and a conversational agent for multi-step analysis (Agent Mode).

### Basic Mode: Standard RAG Pipeline

This mode follows a direct Retrieval-Augmented Generation (RAG) process, ideal for single-turn questions.

1.  **Semantic Retrieval**: The user's query is used to find the most relevant code chunks from the local ChromaDB vector store.
2.  **Contextual Response**: The retrieved code is passed to the LLM along with the original query to generate a direct answer.

```mermaid
graph TD
    A[User Query] --> B[Semantic Search in ChromaDB];
    B --> C[Retrieve Relevant Code Chunks];
    C --> D{LLM};
    A --> D;
    D --> E[Generated Response];
```

### Agent Mode: Conversational Agent

This mode uses a conversational agent that can use tools to answer more complex, multi-step questions. It maintains conversation history and can break down a query into multiple steps.

1.  **Reasoning Loop**: The agent LLM receives the query and conversation history. It decides whether it can answer directly or needs to use a tool (e.g., `search_codebase`).
2.  **Tool Execution**: If a tool is needed, the agent executes it (which may involve a semantic search) and feeds the results back into its context.
3.  **Final Answer**: Once the agent determines it has enough information, it generates the final response.

```mermaid
graph TD
    subgraph Agent
        A[User Query] --> B{Agent LLM};
        B -- "Needs Tool?" --> C[Execute Tool: search_codebase];
        C --> D[Tool Output];
        D --> B;
        B -- "Has Answer" --> E[Final Response];
    end
    subgraph Knowledge Base
        C --> F[Search ChromaDB];
    end
```

---

## Setup Instructions

### 1. Directory Structure Setup
Create the data directory structure for your code projects:
```bash
# Create sibling directory for code projects and vector databases
mkdir ../code-bot-data
mkdir ../code-bot-data/data
mkdir ../code-bot-data/data/my-project1
mkdir ../code-bot-data/data/my-project2
# Vector DBs will be auto-created in ../code-bot-data/vector-db/
```

### 2. Add Your Code Projects
Place your source code projects in the data directory:
```
../code-bot-data/data/
├── spring-boot-app/          # Java/Spring project
│   ├── src/main/java/
│   ├── pom.xml
│   └── application.yml
├── python-service/           # Python project  
│   ├── app/
│   ├── requirements.txt
│   └── config.yaml
└── microservice-config/      # Configuration files
    ├── docker-compose.yml
    └── k8s-manifests/
```

### 3. Download and install Ollama  
   Follow instructions from [Ollama’s official site](https://ollama.com/).

### 4. Download models

   **Full version:**
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

   **Light version:**
   ```bash
   ollama pull llama3.1:8b-instruct-q4_K_M
   ```

### 5. Setup Python environment
   ```bash
   cd C:\dev\github\code-bot
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

### 6. Configure Environment
   The `.env` file is already configured for the new directory structure:

   **For Google AI (Recommended):**
   ```env
   EMBEDDING_PROVIDER=google
   GOOGLE_API_KEY=your_google_api_key_here
   DATA_DIR=../code-bot-data/data
   ```

   **For Ollama (Local):**
   ```env
   EMBEDDING_PROVIDER=ollama
   DATA_DIR=../code-bot-data/data
   ```

   **For OpenAI:**
   ```env
   EMBEDDING_PROVIDER=openai
   OPENAI_API_KEY=sk-your_key_here
   DATA_DIR=../code-bot-data/data
   ```

### 7. Build the code index
   Run the build script to index your source code projects:
   
   **Windows (PowerShell):**
   ```powershell
   $env:EMBEDDING_PROVIDER="google"  # or "ollama", "openai"
   python -m app.build_index
   ```

   **Linux/Mac:**
   ```bash
   export EMBEDDING_PROVIDER=google  # or ollama, openai
   python -m app.build_index
   ```

### 8. Verify the index
   ```bash
   # Example for Google AI index
   python -c "import chromadb; client = chromadb.PersistentClient(path='../code-bot-data/vector-db/chroma_googleai'); print('Collection count:', client.get_collection('local-rag').count())"
   ```

### 9. Run the server
   ```bash
   uvicorn app.main:app --reload
   ```

### 10. Test the health endpoint
    *(No UI required, just the FastAPI server running)*

    **Linux/Mac/PowerShell:**
    ```bash
    curl http://127.0.0.1:8000/health
    ```

    **Windows (PowerShell Native):**
    ```powershell
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
    ```
    # expected output: {"status":"ok"}

### 11. Test code analysis
    
    **Example queries for Java/Spring projects:**
    ```bash
    curl -X POST http://127.0.0.1:8000/chat \
         -H "Content-Type: application/json" \
         -d '{"query":"How does user authentication work in this system?"}'
    ```

    **Example queries for architecture analysis:**
    ```bash
    curl -X POST http://127.0.0.1:8000/chat \
         -H "Content-Type: application/json" \
         -d '{"query":"Trace the request flow from controller to database"}'
    ```

    **Windows (PowerShell):**
    ```powershell
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" `
                     -Method Post `
                     -ContentType "application/json" `
                     -Body '{"query": "What security vulnerabilities exist in the authentication flow?"}'
    ```

### 12. Use Web UI
    ```bash
    cd C:\dev\github\code-bot\local-rag-ui
    npm install
    npm run dev
    ```
    # Go to browser with this URL: http://localhost:5173/

### 13. Stop the app
    ```powershell
    taskkill /F /IM uvicorn.exe /T; taskkill /F /IM python.exe /T; Get-Process | Where-Object { $_.ProcessName -like "*node*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    ```

---

## Switching Providers

The system supports multiple embedding providers. Each provider has its own vector database directory:

- **Google AI**: `../code-bot-data/vector-db/chroma_googleai/` (Recommended)
- **Ollama**: `../code-bot-data/vector-db/chroma_ollama/` (Local/Private)  
- **OpenAI**: `../code-bot-data/vector-db/chroma_openai/` (Alternative cloud)

You must **build the index** for each provider you intend to use, as embeddings are not compatible between providers.

## Example Use Cases

**For Developers:**
- "Find all methods that handle user authentication."
- "Show me an example of how payment processing is implemented."
- "What utility functions are available for date formatting?"
- "Where is the database connection configured in this project?"

**For Architects:**
- "What are the dependencies between these two microservices based on the code?"
- "Show me the main entry points for the `api-gateway` project."
- "How does the system handle logging and error reporting?"

**For Business Stakeholders:**
- "What parts of the code seem to handle customer data?"
- "Can you show me the code related to the checkout process?"
- "Find the configuration for third-party API integrations."

