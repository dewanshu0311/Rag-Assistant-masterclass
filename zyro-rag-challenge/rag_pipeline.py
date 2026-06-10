"""
Zyro Dynamics HR Help Desk — RAG Pipeline
==========================================
A production-grade Retrieval-Augmented Generation pipeline for answering
employee HR questions using Zyro Dynamics internal policy documents.

Architecture:
    User Query → Semantic Guardrail (LLM classification)
               → If HR-relevant: Retrieve top-k chunks → Generate grounded answer
               → If out-of-scope: Return polite refusal

Author: Dewanshu
Competition: NIAT Masterclass RAG Challenge (Kaggle)
"""

import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langsmith import traceable

# ============================================================
# 1. CONFIGURATION
# ============================================================
load_dotenv()

LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"

# Dynamic corpus path resolution to handle both local and Streamlit Community Cloud execution CWDs
_base_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_base_dir)
_path_options = [
    os.path.join(_parent_dir, "zyro-dynamics-hr-corpus"),
    os.path.join(_base_dir, "zyro-dynamics-hr-corpus"),
    os.path.join(os.getcwd(), "zyro-dynamics-hr-corpus"),
    "../zyro-dynamics-hr-corpus/",
    "./zyro-dynamics-hr-corpus/"
]

CORPUS_PATH = os.getenv("CORPUS_PATH")
if not CORPUS_PATH:
    for path in _path_options:
        if os.path.exists(path) and os.path.isdir(path):
            CORPUS_PATH = path
            break
    if not CORPUS_PATH:
        CORPUS_PATH = "../zyro-dynamics-hr-corpus/"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 5
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 512

# ============================================================
# 2. DOCUMENT LOADING
# ============================================================
def load_documents(corpus_path: str = CORPUS_PATH):
    """Load all PDF policy documents from the corpus directory."""
    loader = PyPDFDirectoryLoader(corpus_path, glob="*.pdf")
    documents = loader.load()

    # Enrich metadata: add a clean document title derived from filename
    for doc in documents:
        source = os.path.basename(doc.metadata.get("source", ""))
        # Convert "02_Leave_Policy.pdf" -> "Leave Policy"
        clean_name = re.sub(r"^\d+_", "", source)
        clean_name = clean_name.replace(".pdf", "").replace("_", " ")
        doc.metadata["doc_title"] = clean_name

    print(f"Loaded {len(documents)} document pages from {corpus_path}")
    return documents


# ============================================================
# 3. CHUNKING
# ============================================================
def chunk_documents(documents):
    """Split documents into semantically meaningful chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Prepend document title as context header to each chunk
    for chunk in chunks:
        title = chunk.metadata.get("doc_title", "")
        if title:
            chunk.page_content = f"[Source: {title}]\n{chunk.page_content}"

    print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ============================================================
# 4. EMBEDDINGS
# ============================================================
def init_embeddings():
    """Initialize the HuggingFace embedding model."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"Embedding model initialized: {EMBEDDING_MODEL}")
    return embeddings


# ============================================================
# 5. VECTOR STORE + RETRIEVER
# ============================================================
def build_vectorstore(chunks, embeddings):
    """Build a FAISS vector store from document chunks."""
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"Vector store built with {len(chunks)} vectors")
    return vectorstore


def create_retriever(vectorstore, k: int = RETRIEVER_K):
    """Create a retriever with MMR search for diversity."""
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3},
    )
    print(f"Retriever created (MMR, k={k})")
    return retriever


# ============================================================
# 6. LLM INITIALIZATION
# ============================================================
def init_llm():
    """Initialize the Groq LLM."""
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    print(f"LLM initialized: {LLM_PROVIDER} / {LLM_MODEL}")
    return llm


# ============================================================
# 7. RAG CHAIN
# ============================================================

RAG_TEMPLATE = """You are the official HR Help Desk Assistant for Zyro Dynamics Pvt. Ltd.

IMPORTANT RULES:
1. Answer ONLY based on the provided context from Zyro Dynamics HR policy documents.
2. If the context does not contain enough information to answer, say: "I don't have enough information in the available HR documents to answer this question. Please contact hr.helpdesk@zyrodynamics.com for assistance."
3. Be precise, factual, and cite specific policy details (numbers, dates, durations) when available.
4. The company may be referred to as "Zyro Dynamics", "Acrux Dynamics", or similar variations — treat them all as referring to the same company.
5. Do NOT make up information. Do NOT hallucinate policies that are not in the context.
6. Use bullet points and clear formatting for readability.

CONTEXT FROM HR DOCUMENTS:
{context}

EMPLOYEE QUESTION: {question}

ANSWER (based strictly on the above context):"""

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_TEMPLATE)


def format_docs(docs):
    """Format retrieved documents into a single context string with source attribution."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("doc_title", "Unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"--- Chunk {i} [Source: {title}, Page {int(page)+1}] ---\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


@traceable(name="rag_chain")
def rag_chain(question: str, retriever=None, llm=None):
    """Execute the RAG pipeline: retrieve relevant chunks and generate answer."""
    # Retrieve
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Generate
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "source_documents": docs,
        "context": context,
    }


# ============================================================
# 8. SEMANTIC GUARDRAILS
# ============================================================

OOS_TEMPLATE = """You are a query classifier for an HR Help Desk chatbot at Zyro Dynamics Pvt. Ltd.

Your job is to determine if a user's question is WITHIN SCOPE or OUT OF SCOPE.

WITHIN SCOPE (answer = "IN_SCOPE"):
- Questions about company HR policies (leave, attendance, payroll, benefits, etc.)
- Questions about employee handbook, code of conduct, workplace rules
- Questions about performance reviews, promotions, PIP
- Questions about onboarding, separation, notice period
- Questions about work from home, travel policy, expense claims
- Questions about company profile, leadership, office locations, grade structure
- Questions about IT/data security policies, POSH/harassment policies
- Questions about the company itself (Zyro Dynamics / Acrux Dynamics)

OUT OF SCOPE (answer = "OUT_OF_SCOPE"):
- Questions about competitor companies (Zoho, Freshworks, Salesforce, Google, etc.)
- Questions about financial performance, revenue, stock price, investor data
- Questions about detailed product features, technical product comparisons
- Questions asking for personal opinions, jokes, general knowledge
- Questions completely unrelated to HR or the company
- Questions asking to compare Zyro Dynamics policies with other companies

Respond with EXACTLY one word: either "IN_SCOPE" or "OUT_OF_SCOPE"

Question: {question}
Classification:"""

OOS_PROMPT = ChatPromptTemplate.from_template(OOS_TEMPLATE)

REFUSAL_MESSAGE = (
    "I'm sorry, but this question falls outside the scope of Zyro Dynamics HR policies. "
    "I can only assist with questions related to our internal HR policies, employee benefits, "
    "leave management, performance reviews, workplace guidelines, and similar HR topics. "
    "For other inquiries, please contact the relevant department directly."
)


@traceable(name="guardrail_check")
def check_guardrail(question: str, llm=None) -> bool:
    """Classify whether a question is in-scope for the HR chatbot.
    Returns True if the question is IN_SCOPE, False otherwise.
    """
    chain = OOS_PROMPT | llm | StrOutputParser()
    classification = chain.invoke({"question": question}).strip().upper()
    
    # "IN_SCOPE" is a substring of "OUT_OF_SCOPE", so check "OUT_OF_SCOPE" first
    if "OUT_OF_SCOPE" in classification:
        return False
    return "IN_SCOPE" in classification


@traceable(name="ask_bot")
def ask_bot(question: str, retriever=None, llm=None) -> dict:
    """Main entry point: check guardrails first, then run RAG if in-scope."""
    is_in_scope = check_guardrail(question, llm=llm)

    if not is_in_scope:
        return {
            "answer": REFUSAL_MESSAGE,
            "is_blocked": True,
            "source_documents": [],
        }

    result = rag_chain(question, retriever=retriever, llm=llm)
    result["is_blocked"] = False
    return result


# ============================================================
# 9. PIPELINE INITIALIZATION (one-shot setup)
# ============================================================

_pipeline = {}


def initialize_pipeline(corpus_path: str = CORPUS_PATH):
    """Initialize the full RAG pipeline and cache components."""
    global _pipeline

    if _pipeline:
        print("Pipeline already initialized.")
        return _pipeline

    print("=" * 60)
    print("Initializing Zyro Dynamics HR RAG Pipeline")
    print("=" * 60)

    documents = load_documents(corpus_path)
    chunks = chunk_documents(documents)
    embeddings = init_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    retriever = create_retriever(vectorstore)
    llm = init_llm()

    _pipeline = {
        "documents": documents,
        "chunks": chunks,
        "embeddings": embeddings,
        "vectorstore": vectorstore,
        "retriever": retriever,
        "llm": llm,
    }

    print("=" * 60)
    print("Pipeline ready!")
    print("=" * 60)
    return _pipeline


def query(question: str) -> dict:
    """Convenience function: initialize pipeline if needed and query."""
    if not _pipeline:
        initialize_pipeline()
    return ask_bot(
        question,
        retriever=_pipeline["retriever"],
        llm=_pipeline["llm"],
    )


# ============================================================
# 10. CLI TEST MODE
# ============================================================
if __name__ == "__main__":
    pipeline = initialize_pipeline()

    test_questions = [
        # In-scope HR questions
        "How many days of earned leave do employees get per year?",
        "What is the WFH policy at Zyro Dynamics?",
        "What happens if I take sick leave for 3 consecutive days?",
        # Out-of-scope questions (should be blocked)
        "What was Zyro Dynamics' revenue last year?",
        "Can you compare the leave policy with Zoho?",
        "What are the detailed product features of ZyroCRM vs Salesforce?",
    ]

    for i, q in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Q{i}: {q}")
        print(f"{'='*60}")

        result = ask_bot(
            q,
            retriever=pipeline["retriever"],
            llm=pipeline["llm"],
        )

        if result.get("is_blocked"):
            print(f"[BLOCKED]: {result['answer']}")
        else:
            print(f"[ANSWER]: {result['answer']}")
        print("-" * 60)
        time.sleep(2)  # Rate limit protection
