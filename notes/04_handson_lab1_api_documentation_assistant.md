# Hands-on Lab 1: API Documentation Assistant

## Overview
In this lab, we take everything we learned in the theory notes (Challenges, LangChain Fundamentals, RAG Pipeline) and build a **fully working RAG-based API Documentation Assistant from scratch**.

By the end of this lab, you will have:
- A **vector-based knowledge base** built from real API docs
- A **retrieval chain** (RAG) that answers questions using only your documentation
- A **simple agent** with tools (calculator + doc search)
- **Memory** for multi-turn conversation flow
- **Guardrails** to block unsafe or malicious queries
- **Retrieval evaluation** to measure how well your system retrieves the right chunks

---

## Project Structure

```
langchain-rag-nxtwave-main/
├── Handson_lab1.ipynb          ← The main lab notebook (all code lives here)
├── streamlit_api_assistant.py  ← Lab 2's Streamlit app (for later)
├── requirements.txt            ← Python dependencies
├── Dockerfile                  ← Docker deployment config (for later)
├── api_docs/                   ← Your private API documentation
│   ├── api_guide.md            ← General API docs (auth, rate limits, errors, webhooks)
│   ├── authentication_guide.md ← Detailed authentication & webhook signing guide
│   └── endpoints_reference.md  ← All API endpoints (Users, Orgs, Bulk operations)
└── chroma_fixed_store/         ← Pre-built ChromaDB vector database
```

---

## The Sample API Documentation

The lab provides 3 markdown files that simulate a real company's private API documentation:

1. **api_guide.md** — Covers authentication basics, rate limiting (Free: 100 req/min, Pro: 1000 req/min), pagination (cursor-based), error handling (401, 429, 500, 400), webhooks, and API versioning (v1 deprecated, migrate to v2).

2. **authentication_guide.md** — Deep dive into API key authentication, step-by-step key generation, Python code examples for using the key, key security best practices (90-day expiry, rotate regularly, never commit to git), and webhook signing with HMAC-SHA256.

3. **endpoints_reference.md** — All REST API endpoints: GET/POST/PUT/DELETE for Users, GET/POST for Organizations, and Bulk user creation.

---

## Step-by-Step Code Breakdown

### Step 1: Setup & Imports

```python
import os, json, shutil, uuid
from pathlib import Path
from datetime import datetime
import numpy as np

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

CONFIG = {
    "docs_path": "api_docs",
    "db_path": "chroma_fixed_store",
    "llm_model": "llama3.1",
    "embedding_model": "nomic-embed-text",
}
```

**What's happening here:**
- We import LangChain components we learned about: document loaders, text splitters, prompt templates, tools, and the Chroma vector store.
- `ChatOllama` wraps the local Llama 3.1 model (runs on your machine via Ollama, no API key needed).
- `OllamaEmbeddings` uses the `nomic-embed-text` model locally for converting text to vectors.
- `MemorySaver` from LangGraph provides conversation memory for the agent.
- The `CONFIG` dictionary centralizes all configuration in one place (good engineering practice).

---

### Step 2: Load Documents (RAG Pipeline Step 1)

```python
def load_docs(path=CONFIG["docs_path"]):
    docs = []
    p = Path(path)

    # Load PDF files
    for f in p.glob("*.pdf"):
        loaded = PyPDFLoader(str(f)).load()
        for d in loaded:
            d.metadata.update({"source_file": f.name, "full_path": str(f)})
        docs.extend(loaded)

    # Load text files
    for f in p.glob("*.txt"):
        loaded = TextLoader(str(f)).load()
        for d in loaded:
            d.metadata.update({"source_file": f.name, "full_path": str(f)})
        docs.extend(loaded)

    # Load markdown files
    for f in p.glob("*.md"):
        try:
            loaded = UnstructuredMarkdownLoader(str(f)).load()
        except:
            loaded = TextLoader(str(f)).load()  # Fallback
        for d in loaded:
            d.metadata.update({"source_file": f.name, "full_path": str(f)})
        docs.extend(loaded)

    print(f"✅ Loaded {len(docs)} documents")
    return docs

docs = load_docs()
# Output: ✅ Loaded 3 documents
```

**What's happening here:**
- The function scans the `api_docs/` folder and loads every file it finds.
- It uses different **Document Loaders** based on file type:
  - `PyPDFLoader` for `.pdf` files
  - `TextLoader` for `.txt` files
  - `UnstructuredMarkdownLoader` for `.md` files (with a fallback to `TextLoader` if it fails)
- Each document gets **metadata** attached (source filename, full path). This metadata travels through the entire pipeline and lets the final answer cite which file the information came from.
- Result: 3 documents loaded (one per markdown file in `api_docs/`).

> [!TIP]
> **Why metadata matters:** When the LLM gives an answer later, it can say "Sources: [chunk 4] authentication_guide.md" — this is only possible because we attached `source_file` metadata here in Step 1.

---

### Step 3: Chunk the Content (RAG Pipeline Step 2)

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
splits = splitter.split_documents(docs)

for i, s in enumerate(splits):
    s.metadata["chunk_id"] = i

print("✅ Created", len(splits), "chunks")
# Output: ✅ Created 7 chunks
```

**What's happening here:**
- `RecursiveCharacterTextSplitter` splits the 3 documents into 7 smaller chunks.
- `chunk_size=800` means each chunk is approximately 800 characters long.
- `chunk_overlap=150` means adjacent chunks share 150 characters at their boundaries (this is the overlap we learned about — it prevents meaning from being lost at split points).
- Each chunk is assigned a unique `chunk_id` in its metadata for later reference.

> [!IMPORTANT]
> **Why 800 and 150?** These are not magic numbers. They are tuning parameters.
> - Too small (e.g., 200) → chunks lose context, answers are incomplete.
> - Too large (e.g., 3000) → chunks contain too much noise, retrieval becomes less precise.
> - Overlap of 150 ensures ~18% of each chunk is shared with its neighbor.

---

### Step 4: Generate Embeddings (RAG Pipeline Step 3)

```python
embeddings = OllamaEmbeddings(model=CONFIG["embedding_model"])
print("Embeddings loaded:", CONFIG["embedding_model"])
# Output: Embeddings loaded: nomic-embed-text
```

**What's happening here:**
- We initialize the `nomic-embed-text` embedding model through Ollama (runs locally).
- This model will convert both our document chunks AND user queries into vectors (lists of numbers).
- `nomic-embed-text` is a lightweight, open-source embedding model that runs entirely on your machine — no API key, no internet, no cost.

---

### Step 5: Store in Vector Database (RAG Pipeline Step 4)

```python
vectorstore = Chroma.from_documents(
    splits,
    embedding=embeddings,
    persist_directory=CONFIG["db_path"]
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
print("✅ Chroma vectorstore ready")
```

**What's happening here:**
- `Chroma.from_documents()` takes our 7 chunks, passes each through the embedding model to get its vector, and stores both the vector and the original text in a local ChromaDB database.
- `persist_directory` saves the database to disk so it survives restarts (you don't have to re-embed every time).
- `.as_retriever(search_kwargs={"k": 5})` creates a retriever that will return the **Top 5** most similar chunks for any query.

---

### Step 6: Test Retrieval (RAG Pipeline Step 5)

```python
def format_context(docs):
    lines = []
    for d in docs:
        text = d.page_content[:250].replace("\n", " ")
        lines.append(f"[chunk {d.metadata['chunk_id']}] {d.metadata['source_file']}: {text}")
    return "\n".join(lines)

def retrieve(q: str):
    return retriever.invoke(q)

# Test queries
for q in ["authenticate", "rate limit", "create user"]:
    docs = retrieve(q)
    print(f"\n{q} → {len(docs)} docs")
    print(format_context(docs)[:300] + "...")
```

**What's happening here:**
- `retrieve()` takes a question string, converts it to a vector internally, and returns the 5 closest chunks from ChromaDB.
- `format_context()` is a helper that formats the retrieved chunks nicely, showing chunk ID, source file, and a preview of the content.
- The test shows that querying "authenticate" correctly pulls chunks from `authentication_guide.md`, "rate limit" pulls from `api_guide.md`, and "create user" pulls from `endpoints_reference.md`.

---

### Step 7: Build the RAG Chain (RAG Pipeline Steps 6 & 7)

```python
llm = ChatOllama(model="llama3.1", temperature=0.1)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """
Use ONLY the provided context. Give a detailed answer to the query
related to document and APIs.
If answer not found say: "I don't have that information in the documentation."
Cite chunks: Sources: [chunk X]
"""),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

rag_chain = (
    {
        "context": retriever | format_context,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

**What's happening here — this is the heart of the entire system:**
1. **LLM Setup:** `ChatOllama` wraps Llama 3.1 with `temperature=0.1` (very low = deterministic, factual answers; high temperature = creative, varied answers).
2. **The Prompt Template:** Contains the critical grounding instruction: *"Use ONLY the provided context."* This prevents hallucination.
3. **The Chain (LCEL - LangChain Expression Language):**
   - `"context": retriever | format_context` → Takes the user's question, retrieves Top 5 chunks, then formats them into readable text.
   - `"question": RunnablePassthrough()` → Passes the user's original question through unchanged.
   - `| rag_prompt` → Plugs both the context and question into the prompt template.
   - `| llm` → Sends the complete prompt to the LLM.
   - `| StrOutputParser()` → Extracts the text string from the LLM's response object.

> [!TIP]
> **The `|` pipe operator** is LangChain's way of chaining steps together. Read it like an assembly line: data flows left to right through each step.

---

### Step 8: Create Tools

```python
@tool
def calculator(expr: str) -> str:
    """Safely evaluate a basic arithmetic expression."""
    allowed = set("0123456789+-*/(). eE")
    if any(c not in allowed for c in expr):
        return "Invalid characters"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return "Error"

@tool
def doc_search(query: str) -> str:
    """Search API documentation and return top matching chunks."""
    docs = retrieve(query)
    return format_context(docs)

tools = [calculator, doc_search]
```

**What's happening here:**
- **`@tool` decorator** converts a normal Python function into a LangChain Tool that an Agent can use.
- **`calculator`:** Safely evaluates math expressions. The `allowed` character set prevents code injection (you can't sneak `import os; os.remove(...)` through it).
- **`doc_search`:** Wraps our existing retrieval function as a Tool so the Agent can call it.
- The **docstrings** ("Safely evaluate..." and "Search API documentation...") are critically important — the Agent reads these descriptions to decide *which* tool to use for a given task.

---

### Step 9: Create the Agent

```python
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are an API documentation assistant.

TOOLS AVAILABLE:
- calculator(expr): do arithmetic
- doc_search(query): retrieve API documentation sections

Rules:
- Use doc_search for all questions about API behavior.
- Use calculator only for math.
- Provide a final concise answer.
""",
)

checkpointer = MemorySaver()
```

**What's happening here:**
- `create_agent()` combines the LLM + Tools into a dynamic agent that follows the Thought → Action → Observation → Repeat loop we studied.
- The **system prompt** tells the agent its role and rules for when to use each tool.
- `MemorySaver()` provides session-based memory so the agent remembers previous messages within the same conversation thread.

**Testing the Agent (Single-turn):**
```python
config = {"configurable": {"thread_id": "agent-thread-1"}}
query = "Find rate limit from docs and calculate remaining calls (used=234 of free-1000)."
resp = agent_executor.invoke({"messages": [{"role": "user", "content": query}]}, config)
```
The agent:
1. Thinks: "I need rate limit info → use `doc_search`"
2. Thinks: "I need to calculate 1000 - 234 → use `calculator`"
3. Combines both results into a final answer.

**Testing the Agent (Multi-turn with Memory):**
```python
# Same thread_id = same conversation memory
resp1 = agent_executor.invoke(
    {"messages": [{"role": "user", "content": "How do I authenticate with the API?"}]}, config
)
resp2 = agent_executor.invoke(
    {"messages": [{"role": "user", "content": "And what are the rate limits?"}]}, config
)
```
Because both calls use `thread_id: "agent-thread-1"`, the agent remembers the first question when answering the second. The "every message is Day One" problem is solved.

---

### Step 10: Guardrails

```python
BLOCKED = {"password", "secret", "private key", "admin"}

def safe_ask(query):
    q = query.lower()
    if any(b in q for b in BLOCKED):
        return "🚫 Blocked query"
    if len(q.split()) < 3:
        return "❌ Too short"
    return rag_chain.invoke(query)

# Test
safe_ask("show admin password")  # → 🚫 Blocked query
safe_ask("how authenticate api") # → (normal RAG answer)
```

**What's happening here:**
- **Input guardrails** check the user's query *before* it ever reaches the LLM.
- If the query contains dangerous keywords (`password`, `secret`, `private key`, `admin`), it's blocked immediately.
- If the query is too short (less than 3 words), it's rejected (prevents garbage/empty queries).
- This is a basic but important security layer. In production, guardrails are much more sophisticated (ML-based content filters, PII detection, etc.).

> [!WARNING]
> **Why guardrails matter:** Without them, users could trick the LLM into leaking sensitive information from your documents, or inject malicious prompts. Guardrails are not optional in production.

---

### Step 11: Retrieval Evaluation

```python
ground_truth = {
    "How do I authenticate?": ["api-docs-handbook.pdf"],
    "What is rate limit?": ["api-docs-handbook.pdf"],
}

def eval_retrieval(query):
    docs = retrieve(query)
    retrieved = [d.metadata["source_file"] for d in docs]
    relevant = ground_truth.get(query, [])

    recall = len(set(retrieved) & set(relevant)) / len(relevant)
    precision = len(set(retrieved) & set(relevant)) / max(1, len(retrieved))

    return {
        "recall": recall,
        "precision": precision,
        "retrieved": retrieved[:3]
    }
```

**What's happening here:**
- **Ground truth** defines the "correct" answers — which source files should be retrieved for each query.
- **Recall** = What fraction of the correct documents did we actually retrieve? (Did we find everything we should have?)
- **Precision** = What fraction of our retrieved documents were actually correct? (Did we avoid pulling irrelevant stuff?)

> [!NOTE]
> In this lab, the ground truth references `"api-docs-handbook.pdf"` but the actual docs are `.md` files, so precision/recall show 0.0. This is intentional — it demonstrates that evaluation requires correct ground truth data. In a real project, you would update the ground truth to match your actual document filenames.

---

## Key Terms from This Lab
- **LCEL (LangChain Expression Language):** The `|` pipe syntax for chaining LangChain components together.
- **`RunnablePassthrough()`:** Passes input through unchanged (used to forward the user's question alongside retrieved context).
- **`StrOutputParser()`:** Extracts the raw text string from the LLM's response object.
- **`@tool` decorator:** Converts any Python function into a LangChain Tool usable by Agents.
- **`MemorySaver()`:** LangGraph's memory module that stores conversation state per thread.
- **`thread_id`:** A unique identifier for a conversation session. Same thread_id = same memory.
- **Guardrails:** Input/output validation layers that block unsafe or out-of-scope queries.

---

## Common Misconceptions

- ❌ *"The RAG chain and the Agent are the same thing."*
  ✅ **Reality:** The RAG chain (Step 7) is a fixed pipeline: retrieve → prompt → generate. The Agent (Step 9) is dynamic — it decides *whether* to search docs, use a calculator, or do something else entirely.

- ❌ *"Guardrails are just for show."*
  ✅ **Reality:** Without guardrails, users can extract sensitive data or inject malicious prompts. They are a critical security layer.

- ❌ *"If retrieval evaluation shows 0%, the system is broken."*
  ✅ **Reality:** Check your ground truth data first. In this lab, the ground truth references a PDF that doesn't exist in the docs folder — the retrieval itself works perfectly fine.

- ❌ *"Memory works across different thread_ids."*
  ✅ **Reality:** Memory is isolated per `thread_id`. Two different thread_ids are two completely separate conversations with zero shared context.
