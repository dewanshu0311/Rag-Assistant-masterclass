# Chains, Agents & Advanced RAG

## Overview
Real-world AI assistants need more than just simple retrieval. They need to reason across multiple steps, decide between different tools, rewrite vague user queries, and handle multi-turn conversations gracefully. 

- **Chains** give you speed and predictability.
- **Agents** give you flexibility and reasoning. 
Knowing exactly when to use each is the mark of a great AI engineer.

---

## Chains: When You Know the Steps

A chain is a scripted, step-by-step workflow where you define *exactly* what happens and in what order. Every single run follows the exact same path. Because the path is fixed, chains are extremely fast, easy to debug, and highly reliable.

**Example: The RetrievalQA Chain (The 5 Steps)**
1. Take the user's query
2. Retrieve top 3 relevant chunks from the vector store
3. Format a custom prompt combining those chunks and the query
4. Call the LLM
5. Return the final grounded response

> [!TIP]
> **Key Insight:** When the task is predictable, chains are your most reliable tool. When the task is unpredictable, chains become a limitation—that's when you need agents.

### Types of Chains
- **RetrievalQA Chain:** The backbone of most RAG systems. It simply retrieves context and generates an answer.
- **Router Chains:** Looks at the query and "routes" it to the correct index or prompt. (e.g., sending HR questions to the HR vector store, and IT questions to the IT vector store).
- **Rewriter Chains:** Rewrites vague queries into highly searchable versions *before* retrieval. (e.g., transforming "login not working" into "authentication errors"). Better query = better retrieval.
- **Sequential Chains:** Stacks multiple LLM calls where the output of Step 1 becomes the input of Step 2. (e.g., Step 1: Summarize article -> Step 2: Classify summary sentiment -> Step 3: Write response).

---

## Agents: When the Path Is Unknown

Some user questions cannot be answered in fixed, pre-programmed steps. 

**Example Question:** *"Check the rate limit in the docs and calculate how many requests I can make in 4 hours."*

This requires:
1. Searching docs for the rate limit value.
2. Understanding that quota.
3. Doing math based on that value.
4. Synthesizing the final answer.

You can't write a script for this ahead of time because the math step completely depends on what the search step finds! **Agents use the LLM's own reasoning to decide what to do next.**

### The AgentExecutor Loop
An agent runs in a continuous loop until it solves the problem:
**Thought → Action → Observation → Repeat → Final Answer**

You just give the agent a box of tools, and it figures out how and when to use them.

> [!IMPORTANT]
> **Key Insight:** The *Observation* step is what makes agents intelligent. Without observing the result of an action, the agent would just fire off tools blindly. Observation creates a feedback loop allowing the agent to course-correct if a tool fails.

---

## Tools: What Agents Work With

A tool is simply a Python function wrapped so an agent can call it by name. The agent reads each tool's description and decides which one fits the current task.

**Common Tools:**
- **Calculator:** For arithmetic operations (since LLMs are bad at math).
- **Retriever:** To search the vector store.
- **API Caller:** To fetch live data (e.g., weather, stock prices).
- **Database Lookup:** To run SQL queries.

> [!WARNING]
> **Key Insight:** Tool quality determines agent quality. If your tool description is clear, the agent picks it correctly. If your description is vague, the agent gets confused every single time, regardless of how smart the LLM is.

---

## Agents with Memory

Without memory, an agent suffers from amnesia—every message is a completely fresh start. It cannot refer back to tools it used 2 minutes ago.

**ConversationBufferMemory** fixes this by:
- Letting the agent refer back to earlier steps in the same conversation thread.
- Remembering user preferences.
- Handling complex support queries without losing the plot.

> [!NOTE]
> This transforms a simple "one-shot answer bot" into a true conversational partner. However, remember that `ConversationBufferMemory` resets when the session/thread ends. It does not persist forever.

---

## Chains vs Agents: How to Choose

Chains handle **80%** of RAG use cases. Agents are for the **20%** where flexibility and dynamic decision-making are genuinely required.

> [!TIP]
> **The Golden Rule:** The simplest solution that works reliably is always the right one.

| Scenario | Use What? |
| :--- | :--- |
| Straightforward Q&A from docs | **Chains** |
| Query rewriting before retrieval | **Chains** |
| Deterministic, fixed flows | **Chains** |
| Debugging and tracing pipelines | **Chains** |
| Dynamic tool calling (math + search) | **Agents** |
| Tasks where the path is unknown | **Agents** |

---

## Common Misconceptions

- ❌ *"Agents are always better than chains."*
  ✅ **Reality:** Agents are significantly slower, much harder to debug, and less predictable. Always default to chains unless you absolutely need dynamic reasoning.
  
- ❌ *"You need agents for multi-turn conversations."*
  ✅ **Reality:** A simple Chain + Memory handles almost all multi-turn chat scenarios perfectly.
  
- ❌ *"The agent always picks the right tool."*
  ✅ **Reality:** Agents can and will pick the wrong tools, especially if your tool descriptions are vague.
  
- ❌ *"Sequential chains and agents are the same."*
  ✅ **Reality:** Sequential chains follow a rigid, fixed order (A -> B -> C). Agents decide the order on the fly based on their observations.
