"""
Zyro Dynamics HR Help Desk — RAG Pipeline (V21b)
==================================================
Internal logic synchronized with Kaggle notebook (notebook80936d8ef0).
Public function signatures kept backward-compatible with app.py:

    initialize_pipeline()  — returns pipeline dict with keys:
                             documents, chunks, embeddings, vectorstore, retriever, llm
    ask_bot(question, retriever=None, llm=None)  — main entry point
    check_guardrail(question, llm=None)          — scope classifier
    rag_chain(question, retriever=None, llm=None) — RAG execution

Changed internally:
  - Embedding model  : all-MiniLM-L6-v2  → BAAI/bge-large-en-v1.5
  - Chunk size       : 1000/200           → 1200/250
  - Retrieval        : MMR retriever      → hybrid_retrieve() (dense + lexical)
  - Prompt           : 200-line template  → 14-rule notebook prompt
  - Post-processing  : none              → clean_answer()
  - LLM              : Gemini/Groq        → Groq only (llama-3.3-70b-versatile)
  - Key rotation     : none              → 9-key _rotate_groq_key()
  - Guardrail call   : 3-retry loop      → invoke_prompt_once() with rotation
  - Fallback         : raises exception  → extractive_fallback_answer()

Author: Dewanshu
Competition: NIAT Masterclass RAG Challenge (Kaggle)
"""

import math
import os
import re
import time
from collections import Counter
from typing import Optional

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None
from langsmith import traceable

# ============================================================
# 1. CONFIGURATION
# ============================================================

LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"

# Corpus path: env-var override first, then walk up to find the folder
_base_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_base_dir)
_path_options = [
    os.path.join(_parent_dir, "zyro-dynamics-hr-corpus"),
    os.path.join(_base_dir, "zyro-dynamics-hr-corpus"),
    os.path.join(os.getcwd(), "zyro-dynamics-hr-corpus"),
    "../zyro-dynamics-hr-corpus/",
    "./zyro-dynamics-hr-corpus/",
]
CORPUS_PATH = os.getenv("CORPUS_PATH")
if not CORPUS_PATH:
    for _p in _path_options:
        if os.path.exists(_p) and os.path.isdir(_p):
            CORPUS_PATH = _p
            break
    if not CORPUS_PATH:
        CORPUS_PATH = "../zyro-dynamics-hr-corpus/"

# Notebook-matching parameters
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250

# Module-level globals set by initialize_pipeline()
GROQ_API_KEYS: list = []
CURRENT_GROQ_KEY_INDEX: int = 0
llm = None          # set by initialize_pipeline(); used by invoke_prompt_once()
vectorstore = None  # set by initialize_pipeline(); used by hybrid_retrieve()
chunks: list = []   # set by initialize_pipeline(); used by hybrid_retrieve()
reranker_model = None # set by initialize_pipeline(); used by hybrid_retrieve()

# Cached pipeline dict (keyed so app.py stat cards work)
_pipeline: dict = {}


# ============================================================
# 2. SECRETS LOADING  (Streamlit Cloud → .env fallback)
# ============================================================

def _load_groq_keys() -> list:
    """
    Load Groq API keys.
    Priority:
      1. Streamlit secrets (st.secrets) — used on Streamlit Cloud
      2. Environment variables / .env file — used locally
    The misspelling GROK_KEY_3..9 is intentional — matches Kaggle secret names.
    """
    key_names = [
        "GROQ_KEY_1", "GROQ_KEY_2",
        "GROK_KEY_3", "GROK_KEY_4", "GROK_KEY_5",
        "GROK_KEY_6", "GROK_KEY_7", "GROK_KEY_8", "GROK_KEY_9",
    ]
    keys = []

    # 1. Try Streamlit secrets (only available inside a Streamlit process)
    try:
        import streamlit as st  # noqa: PLC0415
        for name in key_names:
            if name in st.secrets:
                v = str(st.secrets[name]).strip()
                if v:
                    keys.append(v)
        # Single-key fallback
        if not keys and "GROQ_API_KEY" in st.secrets:
            v = str(st.secrets["GROQ_API_KEY"]).strip()
            if v:
                keys.append(v)
    except Exception:
        pass

    # 2. Fall back to environment variables / .env file
    if not keys:
        try:
            from dotenv import load_dotenv  # noqa: PLC0415
            load_dotenv()
        except Exception:
            pass
        for name in key_names:
            v = os.getenv(name, "").strip()
            if v:
                keys.append(v)
        # Single-key fallback
        if not keys:
            v = os.getenv("GROQ_API_KEY", "").strip()
            if v:
                keys.append(v)

    # Deduplicate while preserving order
    seen: set = set()
    deduped = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            deduped.append(k)
    return deduped


def _load_langsmith_key() -> Optional[str]:
    """Load LangSmith API key from Streamlit secrets or environment."""
    try:
        import streamlit as st  # noqa: PLC0415
        if "LANGCHAIN_API_KEY" in st.secrets:
            return str(st.secrets["LANGCHAIN_API_KEY"]).strip() or None
    except Exception:
        pass
    return os.getenv("LANGCHAIN_API_KEY", "").strip() or None


# ============================================================
# 3. LLM — NOTEBOOK-MATCHING FUNCTIONS
# ============================================================

def make_llm():
    """Create an LLM instance based on LLM_PROVIDER."""
    if LLM_PROVIDER == "aicredits":
        api_key = os.environ.get("AICREDITS_API_KEY", "")
        # AICredits provides an OpenAI-compatible endpoint
        base_url = os.environ.get("AICREDITS_BASE_URL", "https://api.aicredits.com/v1") 
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=512,
            api_key=api_key,
            base_url=base_url
        )
    if LLM_PROVIDER == "gemini" and ChatGoogleGenerativeAI is not None:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.0,
            max_output_tokens=512,
            google_api_key=api_key,
        )
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0.0,
        max_tokens=512,
    )


def is_rate_limit_error(error) -> bool:
    """Detect Groq 429 / rate-limit errors (matches notebook Cell 9)."""
    message = str(error).lower()
    return "rate limit" in message or "429" in message or "try again in" in message


def _rotate_groq_key() -> bool:
    """
    Cycle to the next Groq API key (matches notebook Cell 9).
    Returns True if a new key was set, False if all keys are exhausted.
    """
    global CURRENT_GROQ_KEY_INDEX
    CURRENT_GROQ_KEY_INDEX += 1
    if CURRENT_GROQ_KEY_INDEX >= len(GROQ_API_KEYS):
        print("All Groq keys exhausted!")
        return False
    os.environ["GROQ_API_KEY"] = GROQ_API_KEYS[CURRENT_GROQ_KEY_INDEX]
    print(f"Rotated to Groq key {CURRENT_GROQ_KEY_INDEX + 1}/{len(GROQ_API_KEYS)}")
    return True


# ============================================================
# 4. DOCUMENT LOADING
# ============================================================

def load_documents(corpus_path: str = CORPUS_PATH):
    """Load all PDF policy documents from the corpus directory."""
    # Auto-discover corpus path on Kaggle
    if not os.path.exists(corpus_path):
        print(f"Path '{corpus_path}' not found. Auto-discovering...")
        kaggle_input = "/kaggle/input"
        if os.path.exists(kaggle_input):
            for entry in os.listdir(kaggle_input):
                candidate = os.path.join(kaggle_input, entry)
                if os.path.isdir(candidate):
                    # Check if this directory has PDFs
                    import glob as globmod
                    pdfs = globmod.glob(os.path.join(candidate, "**/*.pdf"), recursive=True)
                    if pdfs:
                        corpus_path = candidate
                        print(f"Auto-discovered corpus at: {corpus_path}")
                        break

    loader = PyPDFDirectoryLoader(corpus_path, glob="**/*.pdf")
    documents = loader.load()

    # Fallback: if no PDFs found, try subdirectories
    if not documents and os.path.exists(corpus_path):
        for entry in os.listdir(corpus_path):
            sub = os.path.join(corpus_path, entry)
            if os.path.isdir(sub):
                loader = PyPDFDirectoryLoader(sub, glob="**/*.pdf")
                documents = loader.load()
                if documents:
                    print(f"Found PDFs in subdirectory: {sub}")
                    break

    # Enrich metadata with a clean document title derived from the filename
    for doc in documents:
        source = os.path.basename(doc.metadata.get("source", ""))
        clean_name = re.sub(r"^\d+_", "", source)
        clean_name = clean_name.replace(".pdf", "").replace("_", " ").title()
        doc.metadata["doc_title"] = clean_name

    print(f"Loaded {len(documents)} document pages from {corpus_path}")
    return documents


# ============================================================
# 5. CHUNKING  (notebook: 1200 / 250)
# ============================================================

def chunk_documents(documents):
    """Split documents into semantically meaningful chunks (notebook V4 params)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    _chunks = splitter.split_documents(documents)

    # Prepend document title as context header (notebook Cell 6 pattern)
    for chunk in _chunks:
        source = os.path.basename(chunk.metadata.get("source", "Policy"))
        title = source.replace(".pdf", "").replace("_", " ").title()
        chunk.metadata["doc_title"] = title
        chunk.page_content = f"[Source: {title}]\n{chunk.page_content}"

    print(f"Created {len(_chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return _chunks


# ============================================================
# 6. EMBEDDINGS  (notebook: BAAI/bge-large-en-v1.5, auto-device)
# ============================================================

def init_embeddings():
    """Initialize the HuggingFace embedding model (notebook Cell 7)."""
    try:
        import torch  # noqa: PLC0415
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"Embedding model initialized: {EMBEDDING_MODEL} on {device}")
    return embeddings


# ============================================================
# 7. VECTOR STORE + RETRIEVER
# ============================================================

def build_vectorstore(_chunks, embeddings):
    """Build a FAISS vector store from document chunks."""
    vs = FAISS.from_documents(_chunks, embeddings)
    print(f"Vector store built with {len(_chunks)} vectors")
    return vs


def create_retriever(vs, k: int = 15):
    """
    Create a similarity retriever (k=15).
    Note: hybrid_retrieve() is the primary retrieval path; this retriever is kept
    for app.py compatibility (pipeline['retriever'] stat card).
    """
    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    print(f"Retriever created (similarity, k={k})")
    return retriever


# ============================================================
# 8. NOTEBOOK CELL 10 — RAG CHAIN (V21b)
# ============================================================

# --- Constants ---

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "of", "on",
    "or", "our", "the", "to", "what", "when", "where", "who", "with",
    "you", "your", "does", "do", "tell", "about", "please", "provided",
    "employee", "employees",
}

POLICY_HINTS = {
    "leave": ["Leave Policy"],
    "earned leave": ["Leave Policy"],
    "sick leave": ["Leave Policy"],
    "maternity": ["Leave Policy"],
    "paternity": ["Leave Policy"],
    "salary": ["Compensation And Benefits Policy"],
    "payroll": ["Compensation And Benefits Policy"],
    "ctc": ["Compensation And Benefits Policy"],
    "bonus": ["Compensation And Benefits Policy", "Performance Review Policy"],
    "insurance": ["Compensation And Benefits Policy"],
    "esop": ["Compensation And Benefits Policy", "Onboarding And Separation Policy"],
    "performance": ["Performance Review Policy"],
    "pip": ["Performance Review Policy", "Onboarding And Separation Policy"],
    "apr": ["Performance Review Policy"],
    "promotion": ["Performance Review Policy"],
    "work from home": ["Work From Home Policy"],
    "wfh": ["Work From Home Policy"],
    "remote": ["Work From Home Policy"],
    "job": ["Onboarding And Separation Policy", "Employee Handbook"],
    "recruitment": ["Onboarding And Separation Policy", "Employee Handbook"],
    "hiring": ["Onboarding And Separation Policy", "Employee Handbook"],
    "onboarding": ["Onboarding And Separation Policy"],
    "separation": ["Onboarding And Separation Policy"],
    "notice": ["Onboarding And Separation Policy"],
    "travel": ["Travel And Expense Policy"],
    "expense": ["Travel And Expense Policy"],
    "posh": ["Prevention Of Sexual Harassment Policy"],
    "harassment": ["Prevention Of Sexual Harassment Policy"],
    "security": ["It And Data Security Policy"],
    "password": ["It And Data Security Policy"],
    "founder": ["Company Profile", "Employee Handbook"],
    "leadership": ["Company Profile", "Employee Handbook"],
    "office": ["Company Profile", "Employee Handbook"],
    "grade": ["Company Profile", "Employee Handbook", "Compensation And Benefits Policy"],
}

QUERY_EXPANSIONS = {
    "earned leave": "1.25 days per month 15 days after 1 year 240 days service",
    "maternity": "26 weeks 80 days service expected delivery first two live births",
    "sick leave": "medical certificate more than 2 consecutive days 3 working days",
    "salary": "credited by 7th payroll cut-off 24th following month",
    "payroll": "credited by 7th payroll cut-off 24th following month",
    "ctc": "salary bands grade CTC range bonus target",
    "insurance": "group medical insurance spouse dependent children premiums company paid",
    "pip": "rating 1 or 2 two consecutive review cycles 60 to 90 days",
    "annual performance review": "APR February March April increment promotion letters",
    "work from home": "WFH eligibility hybrid full remote ad-hoc emergency grade L3 L5",
    "wfh": "work from home eligibility hybrid full remote ad-hoc emergency grade L3 L5",
    "job": "onboarding recruitment hiring offer appointment background verification joining documents",
    "recruitment": "hiring offer appointment background verification onboarding joining documents",
    "esop": "employee stock options grade L5 4-year vesting 1-year cliff probation confirmation",
}

# --- Notebook prompt (14 rules, Cell 10) ---

RAG_TEMPLATE = """You are a professional HR Assistant at Zyro Dynamics Pvt. Ltd.
Use the retrieved context to answer the user's question completely and professionally.

CRITICAL RULES:
1. Answer the question fully and naturally in paragraph format. Provide necessary explanations or background details found in the context.
2. Use ONLY information found in the provided context and preserve factual accuracy. Do not fabricate details.
3. NEVER include preamble phrases like "According to...", "Based on...", or "As stated in...". Start your answer directly.
4. NEVER include document names, page numbers, or internal metadata like "Internal Use Only".
5. If the answer is completely absent from the context, output exactly: "The company policies do not explicitly address this."

Context:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_TEMPLATE)


# --- Helper functions (Cell 10) ---

def clean_answer(answer: str) -> str:
    """Strip common LLM filler phrases from the start of the answer."""
    patterns = [
        r"^(According to|Based on|As per|Per the) (the )?(policy|documents?|context)[,.]?\s*",
        r"^(The|This) (policy|document) (states|says|indicates)[,.]?\s*",
        r"^(In|Under) (the )?(leave|salary|insurance|WFH|PIP|APR|ESOP) policy[,.]?\s*",
    ]
    for pat in patterns:
        answer = re.sub(pat, "", answer, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", answer).strip()


def tokenize(text: str):
    """Split text into lowercase non-stopword tokens."""
    return [
        t for t in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if t not in STOPWORDS and len(t) > 1
    ]


def clean_title(doc):
    """Return a clean document title from metadata."""
    title = doc.metadata.get("doc_title")
    if title:
        return title
    return (
        os.path.basename(doc.metadata.get("source", "Policy"))
        .replace(".pdf", "")
        .replace("_", " ")
        .title()
    )


def make_query_variants(question: str):
    """
    Generate up to 3 query variants:
      1. The original question.
      2. Alias-replaced version (Acrux → Zyro).
      3. Keyword-expanded version.
    """
    variants = [question.strip()]
    alias = question
    replacements = [
        ("Acrux Dynamics", "Zyro Dynamics"),
        ("AcruxHR", "ZyroHR"),
        ("AcruxCRM", "ZyroCRM"),
        ("AcruxInsight", "ZyroInsight"),
        ("AcruxDesk", "ZyroDesk"),
        ("Acrux", "Zyro"),
    ]
    for old, new in replacements:
        alias = re.sub(old, new, alias, flags=re.IGNORECASE)
    if alias.lower() != question.lower():
        variants.append(alias)

    lower = question.lower()
    for phrase, expansion in QUERY_EXPANSIONS.items():
        if phrase in lower:
            variants.append(f"{question} {expansion}")
            break

    seen: set = set()
    deduped = []
    for v in variants:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(v)
    return deduped[:3]


def hinted_titles(question: str):
    """Return set of policy document titles matched by query keywords."""
    lower = question.lower()
    titles = []
    for phrase, docs_for_phrase in POLICY_HINTS.items():
        if phrase in lower:
            titles.extend(docs_for_phrase)
    return set(titles)


def lexical_score(question: str, doc) -> float:
    """BM25-like TF score + +10 bonus per matched policy hint title."""
    q_tokens = tokenize(question)
    if not q_tokens:
        return 0.0
    q_counts = Counter(q_tokens)
    text = f"{clean_title(doc)} {doc.page_content}".lower()
    d_counts = Counter(tokenize(text))
    score = 0.0
    for term, weight in q_counts.items():
        if term in d_counts:
            score += (1.0 + math.log1p(d_counts[term])) * weight
    for title in hinted_titles(question):
        if title.lower() in clean_title(doc).lower():
            score += 10.0
    return score


def doc_identity(doc):
    """Return a 3-tuple dedup key for a chunk."""
    return (
        doc.metadata.get("source", ""),
        doc.metadata.get("page", ""),
        doc.page_content[:160],
    )


def hybrid_retrieve(question: str, final_k: int = 15):
    """
    Multi-variant dense retrieval + lexical reranking (notebook Cell 10).
    Uses module-level `vectorstore` and `chunks` set by initialize_pipeline().
    """
    candidates = []
    seen: set = set()

    # Dense retrieval across up to 3 query variants
    for variant in make_query_variants(question):
        for rank, doc in enumerate(vectorstore.similarity_search(variant, k=15), 1):
            ident = doc_identity(doc)
            if ident not in seen:
                seen.add(ident)
                candidates.append((doc, 20.0 - rank))

    # Lexical top-10 additions (BM25-like)
    lexical_ranked = sorted(
        ((doc, lexical_score(question, doc)) for doc in chunks),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    for doc, score in lexical_ranked:
        if score <= 0:
            continue
        ident = doc_identity(doc)
        if ident not in seen:
            seen.add(ident)
            candidates.append((doc, score))

    # Rerank: combined score (lexical + rank_score), break ties by rank_score
    reranked = sorted(
        candidates,
        key=lambda item: (lexical_score(question, item[0]) + item[1], item[1]),
        reverse=True,
    )
    top_candidates = [doc for doc, _ in reranked[:final_k]]
    
    if reranker_model:
        scores = reranker_model.predict([(question, doc.page_content) for doc in top_candidates])
        scored_candidates = sorted(zip(top_candidates, scores), key=lambda x: x[1], reverse=True)
        best_score = scored_candidates[0][1]
        final_docs = []
        for doc, score in scored_candidates:
            if len(final_docs) < 2:
                final_docs.append(doc)
            elif len(final_docs) < 4 and (best_score - score) <= 2.5:
                final_docs.append(doc)
        return final_docs
        
    return top_candidates


def format_docs(docs):
    """Format retrieved chunks into a single context string with source labels."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        title = clean_title(doc)
        page = doc.metadata.get("page", "?")
        try:
            page_label = int(page) + 1
        except Exception:
            page_label = "N/A"
        formatted.append(f"[Source: {title}, Page {page_label}, Chunk {i}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def invoke_prompt_once(prompt, payload: dict, label: str = "LLM call") -> str:
    """
    Single LLM call with one automatic key-rotation retry on 429 (notebook Cell 10).
    Uses and mutates the module-level `llm` global.
    """
    global llm
    try:
        chain = prompt | llm | StrOutputParser()
        return chain.invoke(payload)
    except Exception as first_error:
        if LLM_PROVIDER == "groq" and is_rate_limit_error(first_error) and _rotate_groq_key():
            print(f"{label} hit a rate limit. Rotating Groq key and retrying once...")
            llm = make_llm()
            chain = prompt | llm | StrOutputParser()
            return chain.invoke(payload)
        raise first_error


def split_candidate_sentences(text: str):
    """Split text into sentences >= 20 chars for extractive fallback."""
    text = re.sub(r"\s+", " ", text).strip()
    pieces = re.split(r"(?<=[.!?])\s+|\n", text)
    return [p.strip(" -;:") for p in pieces if len(p.strip()) > 20]


def is_table_like(text: str) -> bool:
    """Return True if the extracted text is mostly numeric (looks like a table)."""
    lines = text.split("\n")
    if len(lines) > 3:
        num_lines_with_numbers = sum(1 for line in lines if re.search(r"\d", line))
        if num_lines_with_numbers / len(lines) > 0.6:
            return True
    return False


def extractive_fallback_answer(question: str, docs) -> str:
    """
    Extract the best matching sentences directly from retrieved chunks.
    Used when the LLM call fails entirely (notebook Cell 10).
    """
    q_terms = set(tokenize(question))
    scored = []
    for doc in docs[:6]:
        title = clean_title(doc)
        for sentence in split_candidate_sentences(doc.page_content):
            s_terms = set(tokenize(sentence))
            overlap = len(q_terms & s_terms)
            bonus = lexical_score(question, doc) * 0.05
            score = overlap + bonus
            if score > 0:
                scored.append((score, title, sentence))

    scored.sort(reverse=True, key=lambda item: item[0])
    selected = []
    seen: set = set()
    for _, title, sentence in scored:
        key = sentence[:100].lower()
        if key in seen:
            continue
        seen.add(key)
        if len(sentence) <= 200:
            selected.append(f"According to {title}, {sentence}")
        if len(" ".join(selected)) > 1000 or len(selected) >= 3:
            break

    if not selected and docs:
        title = clean_title(docs[0])
        excerpt = re.sub(r"\s+", " ", docs[0].page_content).strip()[:1200]
        selected.append(f"According to {title}, {excerpt}")

    if not selected:
        return (
            "I don't have enough information in the available HR documents to answer this question. "
            "Please contact hr.helpdesk@zyrodynamics.com for assistance."
        )

    fallback = " ".join(selected)
    if is_table_like(fallback):
        return (
            "I don't have enough information in the available HR documents to answer this question precisely. "
            "Please contact hr.helpdesk@zyrodynamics.com for assistance."
        )
    return fallback


@traceable(name="rag_chain_v21")
def rag_chain(question: str, retriever=None, llm=None) -> dict:
    """
    Execute the RAG pipeline: hybrid retrieve → LLM generate → clean → return.

    `retriever` and `llm` kwargs are accepted for app.py backward-compatibility
    but are not used internally; the module-level globals are used instead.
    """
    docs = hybrid_retrieve(question, final_k=15)
    context = format_docs(docs)

    try:
        answer = invoke_prompt_once(
            RAG_PROMPT,
            {"context": context, "question": question},
            label="Answer generation",
        )
        answer = clean_answer(answer)
        answer_source = "llm"
    except Exception as e:
        print(f"Answer generation failed; using extractive fallback. Error: {e}")
        answer = extractive_fallback_answer(question, docs)
        answer_source = "extractive_fallback"

    return {
        "answer": answer,
        "answer_source": answer_source,
        "source_documents": docs,
        "context": context,
    }


# ============================================================
# 9. NOTEBOOK CELL 11 — GUARDRAILS (V21 Local-First)
# ============================================================

CLEAR_OOS_PATTERNS = [
    r"\b(revenue|stock price|valuation|funding|investor|profit margin|financial performance|company performing financially)\b",
    r"\b(zoho|freshworks|salesforce|google|microsoft|amazon|oracle|sap)\b",
    r"\b(compare|comparison|versus|vs\.?|better than)\b.*\b(company|policy|policies|product|salesforce|zoho|freshworks)\b",
    r"\b(write|generate|create|debug)\b.*\b(code|script|program|python|sql|javascript)\b",
    r"\b(joke|weather|movie|recipe|capital of|general knowledge)\b",
]

PRODUCT_DETAIL_PATTERN = re.compile(
    r"\b(detailed\s+)?(product\s+)?features?\b.*\b(acruxcrm|zyrocrm|acruxhr|zyrohr|acruxinsight|zyroinsight|acruxdesk|zyrodesk)\b|"
    r"\b(acruxcrm|zyrocrm|acruxhr|zyrohr|acruxinsight|zyroinsight|acruxdesk|zyrodesk)\b.*\b(detailed\s+)?(product\s+)?features?\b",
    re.IGNORECASE,
)

IN_SCOPE_HINTS = [
    "zyro", "acrux", "hr", "leave", "salary", "payroll", "ctc", "bonus",
    "insurance", "benefit", "performance", "pip", "promotion", "onboarding",
    "separation", "notice period", "wfh", "work from home", "travel", "expense",
    "reimbursement", "posh", "harassment", "it security", "data security", "grade",
    "probation", "attendance", "recruitment", "job", "hiring", "esop",
    "founder", "office", "leadership",
]

OOS_TEMPLATE = """You are a query classifier for an HR Help Desk chatbot at Zyro Dynamics Pvt. Ltd.
The company may be called Zyro Dynamics or Acrux Dynamics.
Classify the user's question as IN_SCOPE or OUT_OF_SCOPE.

IN_SCOPE:
- HR policies, benefits, leave, attendance, payroll, compensation, performance, PIP, promotions
- onboarding, separation, notice period, WFH, travel, expenses, IT/data security, POSH
- company profile facts present in internal documents, such as founders, offices, leadership, products list, and grade structure

OUT_OF_SCOPE:
- competitor companies or competitor policy comparisons
- financial performance, revenue, valuation, funding, stock price, investor data
- detailed product features or comparisons with external products such as Salesforce
- coding, math, jokes, general knowledge, or unrelated requests

Respond with EXACTLY one word: IN_SCOPE or OUT_OF_SCOPE

Question: {question}

Classification:"""

OOS_PROMPT = ChatPromptTemplate.from_template(OOS_TEMPLATE)

REFUSAL_MESSAGE = (
    "I'm sorry, but this question falls outside the scope of Zyro Dynamics HR policies and "
    "internal company documents. I can assist with Zyro/Acrux HR policies, benefits, leave, "
    "compensation, performance reviews, workplace guidelines, onboarding, travel, IT security, "
    "POSH, and company profile information available in the corpus. For other inquiries, please "
    "contact the relevant department directly."
)


def local_guardrail_decision(question: str) -> Optional[bool]:
    """
    Fast local guardrail (no LLM call).
    Returns True (in-scope), False (out-of-scope), or None (needs LLM).
    """
    text = question.lower()
    if PRODUCT_DETAIL_PATTERN.search(question):
        return False
    for pattern in CLEAR_OOS_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return False
    if any(hint in text for hint in IN_SCOPE_HINTS):
        return True
    return None


@traceable(name="guardrail_check_v21")
def check_guardrail(question: str, llm=None) -> bool:
    """
    Classify whether a question is in-scope for the HR chatbot.
    Returns True if IN_SCOPE, False otherwise.

    `llm` kwarg is accepted for app.py backward-compatibility but is not used
    internally; invoke_prompt_once() uses the module-level llm global.
    """
    local_decision = local_guardrail_decision(question)
    if local_decision is not None:
        return local_decision

    try:
        classification = invoke_prompt_once(
            OOS_PROMPT,
            {"question": question},
            label="Guardrail classification",
        ).strip().upper()
    except Exception as e:
        print(f"Guardrail classifier unavailable; defaulting to OUT_OF_SCOPE. Error: {e}")
        return False

    # "IN_SCOPE" is a substring of "OUT_OF_SCOPE", so check OUT_OF_SCOPE first
    if "OUT_OF_SCOPE" in classification:
        return False
    return "IN_SCOPE" in classification


@traceable(name="ask_bot_v21")
def ask_bot(question: str, retriever=None, llm=None) -> dict:
    """
    Main entry point: check guardrails, then run RAG if in-scope.

    `retriever` and `llm` kwargs are accepted for app.py backward-compatibility
    but are not used internally.
    """
    is_in_scope = check_guardrail(question)

    if not is_in_scope:
        return {
            "answer": REFUSAL_MESSAGE,
            "answer_source": "guardrail_refusal",
            "is_blocked": True,
            "source_documents": [],
        }

    result = rag_chain(question)
    result["is_blocked"] = False
    return result


# ============================================================
# 10. PIPELINE INITIALIZATION (one-shot setup)
# ============================================================

def initialize_pipeline(corpus_path: str = CORPUS_PATH) -> dict:
    """
    Initialize the full RAG pipeline and cache all components.
    Called once by app.py via @st.cache_resource.

    Returns a dict with keys:
        documents, chunks, embeddings, vectorstore, retriever, llm
    (all keys required by app.py stat cards and ask_bot call)
    """
    global _pipeline, vectorstore, chunks, llm, GROQ_API_KEYS, CURRENT_GROQ_KEY_INDEX, reranker_model

    if _pipeline:
        print("Pipeline already initialized.")
        return _pipeline

    print("=" * 60)
    print("Initializing Zyro Dynamics HR RAG Pipeline (V21b)")
    print("=" * 60)

    # --- Secrets (loaded lazily here, not at import time) ---
    GROQ_API_KEYS = _load_groq_keys()
    CURRENT_GROQ_KEY_INDEX = 0
    if GROQ_API_KEYS:
        os.environ["GROQ_API_KEY"] = GROQ_API_KEYS[0]
        print(f"Loaded {len(GROQ_API_KEYS)} Groq API key(s).")
    else:
        print("WARNING: No Groq API keys found. LLM calls will fail.")

    langsmith_key = _load_langsmith_key()
    if langsmith_key:
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGCHAIN_PROJECT", "zyro-rag-challenge-v21")

    # --- Pipeline components ---
    documents = load_documents(corpus_path)
    chunks = chunk_documents(documents)          # sets module-level `chunks`
    embeddings = init_embeddings()
    
    if reranker_model is None:
        print("Loading CrossEncoder reranker ('cross-encoder/ms-marco-MiniLM-L-6-v2')...")
        from sentence_transformers import CrossEncoder
        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(BASE_DIR, "faiss_index")

    if os.path.exists(index_path):
        print(f"Loading prebuilt FAISS index from '{index_path}'...")
        try:
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            print("FAISS index loaded successfully.")
        except Exception as e:
            print(f"FAISS.load_local() FAILED with exception: {e}")
            print("Falling back to building FAISS index from scratch...")
            vectorstore = build_vectorstore(chunks, embeddings)
            vectorstore.save_local(index_path)
    else:
        print("Building FAISS index...")
        vectorstore = build_vectorstore(chunks, embeddings)
        print(f"Saving FAISS index to '{index_path}'...")
        vectorstore.save_local(index_path)

    retriever = create_retriever(vectorstore)
    llm = make_llm()                             # sets module-level `llm`

    _pipeline = {
        "documents": documents,
        "chunks": chunks,
        "embeddings": embeddings,
        "vectorstore": vectorstore,
        "retriever": retriever,
        "llm": llm,
        "reranker_model": reranker_model,
    }

    print("=" * 60)
    print("Pipeline ready! (V21b)")
    print("=" * 60)
    return _pipeline


def query(question: str) -> dict:
    """Convenience CLI wrapper: initialize pipeline if needed, then query."""
    if not _pipeline:
        initialize_pipeline()
    return ask_bot(question)


# ============================================================
# 11. CLI TEST MODE
# ============================================================
if __name__ == "__main__":
    pipeline = initialize_pipeline()

    test_questions = [
        # In-scope HR questions
        "How many days of earned leave do employees get per year?",
        "What is the WFH policy at Zyro Dynamics?",
        "What happens if I take sick leave for 3 consecutive days?",
        "When is salary credited each month?",
        # Out-of-scope (should be blocked)
        "What was Zyro Dynamics' revenue last year?",
        "Can you compare the leave policy with Zoho?",
    ]

    for i, q in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Q{i}: {q}")
        print(f"{'='*60}")
        result = ask_bot(q)
        if result.get("is_blocked"):
            print(f"[BLOCKED]: {result['answer']}")
        else:
            print(f"[ANSWER]: {result['answer']}")
            print(f"[SOURCE]: {result.get('answer_source', '')}")
        print("-" * 60)
        time.sleep(2)
