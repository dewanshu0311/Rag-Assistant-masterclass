# Hands-on Lab 2: Streamlit App & Deployment

## Overview
In Lab 1, we built the RAG pipeline in a Jupyter notebook (backend only — no user interface). In Lab 2, we take that same pipeline and wrap it in a **Streamlit web application** — a real, interactive UI where users can type questions, see answers, inspect retrieved chunks, test guardrails, run the agent, and view LangSmith traces. Then we learn how to deploy it live to the internet.

---

## What the Streamlit App Does

The file [streamlit_api_assistant.py](file:///c:/Rag%20Assistant%20masterclass/langchain-rag-nxtwave-main/streamlit_api_assistant.py) is a **single-file application** (639 lines) that combines everything from Lab 1 into a professional web interface with 6 tabs:

1. **RAG Q&A** — Type a question, get an answer grounded in your API docs, plus see the retrieved chunks.
2. **Agent** — Run the agent with tools (calculator + doc search) and memory.
3. **Guardrails** — Test whether a query gets blocked or passes the safety filter.
4. **Retrieved Chunks** — Inspect exactly which document chunks the retriever pulls for any query.
5. **Evaluation / Retrieval Tests** — Run Recall@K and Precision@K tests against ground-truth data.
6. **LangSmith Traces** — View links to LangSmith trace dashboard for every run.

---

## How the Code Maps to What We Already Know

Every section of this Streamlit app is something we already built in Lab 1 — just wrapped in a UI:

| Lab 1 (Notebook) | Lab 2 (Streamlit App) |
| :--- | :--- |
| `load_docs()` function | `load_documents()` with `@st.cache_resource` decorator |
| `RecursiveCharacterTextSplitter` | `build_vectorstore()` function |
| `Chroma.from_documents()` | Same, but triggered by "Initialize / Rebuild Index" button |
| `rag_chain` (LCEL pipe) | `build_rag_chain()` function |
| `@tool calculator` | `calculator_tool()` with enhanced error handling |
| `@tool doc_search` | `doc_search_tool()` that reads from `st.session_state` |
| `create_agent()` | `create_tool_agent()` function |
| `BLOCKED` set + `safe_ask()` | `apply_guardrails()` with expanded blocked keywords |
| `eval_retrieval()` | `retrieval_metrics()` with JSON ground-truth input |
| (not in Lab 1) | LangSmith trace capture and dashboard |

> [!TIP]
> **Key Insight:** The Streamlit app doesn't add new AI concepts. It's the exact same RAG + Agent + Guardrails + Evaluation pipeline from Lab 1, just with a web interface bolted on top.

---

## Code Breakdown: What's New in Lab 2

### 1. Streamlit Basics
```python
import streamlit as st

st.set_page_config(page_title="API Docs Assistant", layout="wide")
st.title("📘 API Documentation Assistant")
```
- `st.set_page_config()` sets the browser tab title and layout.
- `st.title()` displays the main heading.
- `st.sidebar` creates the left sidebar for configuration.
- `st.tabs()` creates the 6 tabbed sections.

### 2. Session State
```python
if "index_built" not in st.session_state:
    st.session_state.index_built = False
```
Streamlit reruns the entire script on every user interaction (button click, text input). `st.session_state` is how you persist data across reruns — it's Streamlit's version of "memory."

### 3. Caching
```python
@st.cache_resource
def load_documents(docs_path: str):
    ...
```
`@st.cache_resource` tells Streamlit: "Run this function once and cache the result. Don't re-run it on every page refresh." This prevents re-loading documents and re-building the vector store every time a user clicks a button.

### 4. The Sidebar (Configuration Panel)
The sidebar lets users configure:
- Docs folder path
- ChromaDB path
- LLM model name
- Embedding model name
- Chunk size and overlap
- Retrieval K value
- LangSmith API key and project name

### 5. Enhanced Guardrails
```python
BLOCKED_KEYWORDS = {"password", "secret", "private key", "admin", "token", "ssn"}
```
Lab 2 expands the blocked keywords list to include "token" and "ssn" (Social Security Number) on top of what Lab 1 had.

### 6. LangSmith Integration
```python
if enable_langsmith and ls_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = ls_key
    os.environ["LANGCHAIN_PROJECT"] = ls_project
```
When enabled, every RAG and Agent run is automatically traced by LangSmith. The app tries to find the latest trace URL and displays it in the "LangSmith Traces" tab.

---

## How to Run the App Locally

### Prerequisites
1. **Ollama must be running** (it provides the local LLM and embedding model):
   ```bash
   ollama serve
   ```
   And make sure you've pulled the required models:
   ```bash
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```

2. **Install Python dependencies:**
   ```bash
   pip install streamlit langchain-ollama langchain-chroma langchain-community chromadb langsmith pypdf langgraph
   ```

### Running
```bash
cd c:\Rag Assistant masterclass\langchain-rag-nxtwave-main
streamlit run streamlit_api_assistant.py
```
This opens the app at `http://localhost:8501` in your browser.

### First-Time Setup in the App
1. Click **"Initialize / Rebuild Index"** button in the sidebar to build the vector store from the `api_docs/` folder.
2. Once the index is built, all 6 tabs become functional.
3. Go to the **RAG Q&A** tab and ask: *"How do I authenticate with the API?"*

---

## Deployment Options

### Option 1: Run Locally Only (Simplest)
Just run `streamlit run streamlit_api_assistant.py` on your machine. No deployment needed. This is what most students do for the masterclass.

### Option 2: Deploy to Streamlit Cloud (Free, Public URL)
1. Push your code to a **GitHub repository** (but do NOT push `.env`).
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub.
3. Click "New App" → Select your repo → Set main file to `streamlit_api_assistant.py` → Deploy.
4. Add your API keys in the app's **Settings → Secrets** section (not in code).
5. You get a live public URL like `https://your-app.streamlit.app`.

> [!WARNING]
> **Important Limitation:** This app uses **Ollama** (local LLM), which does NOT work on Streamlit Cloud because Streamlit Cloud doesn't have Ollama installed. To deploy to the cloud, you would need to switch the code to use an API-based LLM (like OpenAI's GPT or Google's Gemini) instead of `ChatOllama`.

### Option 3: Docker Deployment
The included `Dockerfile` packages the app into a container. Same Ollama limitation applies — Docker deployment would need either Ollama running alongside or an API-based LLM.

---

## Key Terms from This Lab
- **Streamlit:** A Python framework that turns scripts into interactive web apps with minimal code.
- **`st.session_state`:** Streamlit's mechanism for persisting data across page reruns.
- **`@st.cache_resource`:** Decorator that caches expensive function results (like loading documents or building vector stores).
- **Streamlit Cloud:** Free hosting service for Streamlit apps, connected directly to GitHub repos.
- **Secrets Management:** Storing API keys in Streamlit Cloud's secure secrets panel instead of hardcoding them.

---

## Common Misconceptions

- ❌ *"I need to understand frontend web development (HTML/CSS/JavaScript) to build this."*
  ✅ **Reality:** Streamlit handles all the frontend. You write only Python. The `st.text_input()`, `st.button()`, `st.tabs()` functions generate the entire UI automatically.

- ❌ *"I must deploy the app to complete the lab."*
  ✅ **Reality:** Running locally with `streamlit run` is perfectly sufficient. Deployment is an optional bonus step.

- ❌ *"The Streamlit app is a completely different codebase from Lab 1."*
  ✅ **Reality:** It's the exact same RAG pipeline, agent, guardrails, and evaluation logic — just wrapped in Streamlit UI functions.

- ❌ *"I need to pay for hosting."*
  ✅ **Reality:** Streamlit Cloud is completely free for public apps. The only cost is the LLM API key if you switch from Ollama to OpenAI/Gemini.
