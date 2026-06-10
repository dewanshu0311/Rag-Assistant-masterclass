"""
Zyro Dynamics HR Help Desk — Streamlit Application
====================================================
A premium, dark-themed chatbot interface for the HR RAG pipeline.

Author: Dewanshu
Competition: NIAT Masterclass RAG Challenge (Kaggle)
"""

import streamlit as st
import os
import sys
import time

# ── Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS Styling ─────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* ── Glassmorphism Chat Container ── */
.chat-container {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.8), rgba(10, 14, 26, 0.9));
    backdrop-filter: blur(20px);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

/* ── Answer Card ── */
.answer-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.05));
    border-left: 4px solid #6366f1;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
    color: #e2e8f0;
    line-height: 1.7;
    font-size: 15px;
}

/* ── Blocked Card ── */
.blocked-card {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(220, 38, 38, 0.05));
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
    color: #fca5a5;
    line-height: 1.7;
    font-size: 15px;
}

/* ── Source Chip ── */
.source-chip {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 20px;
    padding: 4px 14px;
    margin: 4px 4px;
    font-size: 12px;
    color: #a5b4fc;
    font-weight: 500;
}

/* ── Hero Header ── */
.hero-header {
    text-align: center;
    padding: 30px 0 10px 0;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    font-weight: 400;
}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629, #0a0e1a) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.1);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    color: #cbd5e1 !important;
}

/* ── Stats Cards ── */
.stat-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.05));
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 8px 0;
}

.stat-number {
    font-size: 1.8rem;
    font-weight: 700;
    color: #a78bfa;
}

.stat-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
    margin: 20px 0;
}

/* ── Hide Streamlit Default Elements ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Initialize RAG Pipeline ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Initialize the RAG pipeline once and cache it."""
    from rag_pipeline import initialize_pipeline
    return initialize_pipeline()


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏢 Zyro Dynamics")
    st.markdown("**HR Help Desk Assistant**")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 📋 Available Topics")
    topics = [
        "Leave & Attendance Policy",
        "Work From Home Policy",
        "Compensation & Benefits",
        "Performance Reviews & PIP",
        "Code of Conduct",
        "IT & Data Security",
        "POSH / Harassment Policy",
        "Onboarding & Separation",
        "Travel & Expense Policy",
        "Company Profile & Structure",
    ]
    for topic in topics:
        st.markdown(f"- {topic}")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 💡 Sample Questions")
    sample_qs = [
        "How many days of earned leave do I get?",
        "What is the WFH eligibility?",
        "When is salary credited each month?",
        "What is the PIP duration?",
        "Who founded Zyro Dynamics?",
    ]
    for q in sample_qs:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state["prefill_question"] = q

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center; color:#64748b; font-size:12px;'>"
        "Built with LangChain + Groq + FAISS<br>"
        "RAG Challenge 2025"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main Content ─────────────────────────────────────────────

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">Zyro Dynamics HR Help Desk</div>
    <div class="hero-subtitle">Ask anything about company policies, benefits, leave, and more</div>
</div>
<div class="custom-divider"></div>
""", unsafe_allow_html=True)


# Load pipeline
with st.spinner("🔄 Initializing HR Knowledge Base..."):
    pipeline = load_pipeline()

# Stats Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-number">{len(pipeline['documents'])}</div>
        <div class="stat-label">Pages Loaded</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-number">{len(pipeline['chunks'])}</div>
        <div class="stat-label">Knowledge Chunks</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="stat-card">
        <div class="stat-number">11</div>
        <div class="stat-label">Policy Documents</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""<div class="stat-card">
        <div class="stat-number">✓</div>
        <div class="stat-label">Guardrails Active</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Chat Interface
st.markdown("### 💬 Ask Your HR Question")

# Handle prefilled questions from sidebar
prefill = st.session_state.pop("prefill_question", "")

question = st.text_input(
    "Type your question below:",
    value=prefill,
    placeholder="e.g., How many days of earned leave can I carry forward?",
    label_visibility="collapsed",
)

col_ask, col_clear = st.columns([1, 5])
with col_ask:
    ask_clicked = st.button("🔍 Ask HR Bot", type="primary", use_container_width=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process question
if ask_clicked and question.strip():
    with st.spinner("🤖 Searching HR policies and generating answer..."):
        from rag_pipeline import ask_bot

        start_time = time.time()
        result = ask_bot(
            question,
            retriever=pipeline["retriever"],
            llm=pipeline["llm"],
        )
        elapsed = time.time() - start_time

    # Store in history
    st.session_state.chat_history.insert(0, {
        "question": question,
        "answer": result["answer"],
        "is_blocked": result.get("is_blocked", False),
        "sources": result.get("source_documents", []),
        "time": elapsed,
    })

# Display chat history
for entry in st.session_state.chat_history:
    st.markdown(f"""<div class="chat-container">
        <div style="color:#a5b4fc; font-weight:600; font-size:14px; margin-bottom:8px;">
            🧑‍💼 Employee Question
        </div>
        <div style="color:#e2e8f0; font-size:15px; margin-bottom:16px;">{entry['question']}</div>
    </div>""", unsafe_allow_html=True)

    if entry["is_blocked"]:
        st.markdown(f"""<div class="blocked-card">
            <div style="font-weight:600; margin-bottom:8px;">🚫 Out of Scope</div>
            {entry['answer']}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="answer-card">
            <div style="font-weight:600; margin-bottom:8px; color:#a5b4fc;">
                🤖 HR Assistant Response
            </div>
            {entry['answer']}
        </div>""", unsafe_allow_html=True)

        # Show source documents
        if entry.get("sources"):
            seen_titles = set()
            source_chips = ""
            for doc in entry["sources"]:
                title = doc.metadata.get("doc_title", "Unknown")
                if title not in seen_titles:
                    seen_titles.add(title)
                    source_chips += f'<span class="source-chip">📄 {title}</span>'

            st.markdown(f"""<div style="margin-top:8px;">
                <span style="color:#64748b; font-size:12px;">Sources: </span>
                {source_chips}
            </div>""", unsafe_allow_html=True)

            with st.expander("📚 View Retrieved Chunks"):
                for i, doc in enumerate(entry["sources"], 1):
                    title = doc.metadata.get("doc_title", "Unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i}** — {title} (Page {int(page)+1})")
                    st.code(doc.page_content[:500], language=None)

    # Response time
    st.markdown(
        f'<div style="text-align:right; color:#475569; font-size:11px;">'
        f'⚡ Response time: {entry["time"]:.2f}s</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
