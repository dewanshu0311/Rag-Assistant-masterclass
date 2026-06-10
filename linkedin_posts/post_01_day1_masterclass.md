# LinkedIn Post #1 — NIAT RAG Masterclass Day 1

## 📝 Text Post (Copy-paste this as your LinkedIn caption)

---

Started the NIAT LangChain Masterclass today — "Building a Production-Ready RAG Assistant" by Aakriti Agarwal (Senior AI Engineer @ IBM).

Before today, I thought building an AI chatbot was mostly about picking the right model. GPT-4, Claude, Gemini — just plug it in and it works, right?

Turns out, the model is actually the easiest part. The hard part is everything around it.

Here's what hit me today:

𝟭. 𝗟𝗟𝗠𝘀 𝗱𝗼𝗻'𝘁 𝗿𝗲𝗺𝗲𝗺𝗯𝗲𝗿 𝗮𝗻𝘆𝘁𝗵𝗶𝗻𝗴.
Every message you send is Day 1 for the model. That "memory" you see in ChatGPT? It's engineered separately. The LLM itself is completely stateless.

I dug deeper into this and found that the simplest memory approach (ConversationBufferMemory) literally re-sends your entire chat history with every single message. Which means if you've chatted for 2 hours, even a simple "hi" costs you thousands of tokens. That's expensive and breaks the context window fast.

𝟮. 𝗟𝗟𝗠𝘀 𝗰𝗮𝗻'𝘁 𝗱𝗼 𝗺𝗮𝘁𝗵.
This sounds insane — they run on supercomputers. But LLMs don't calculate. They predict the next word. When you ask "what's 2+2?", it doesn't compute 4. It guesses "4" because it has seen "2+2=4" millions of times in training data. Ask it something it hasn't seen, and it confidently gives you the wrong number.

That's why production AI apps give LLMs a calculator tool instead of making them do math.

𝟯. 𝗥𝗔𝗚 = 𝗥𝗲𝘁𝗿𝗶𝗲𝘃𝗮𝗹-𝗔𝘂𝗴𝗺𝗲𝗻𝘁𝗲𝗱 𝗚𝗲𝗻𝗲𝗿𝗮𝘁𝗶𝗼𝗻
Instead of hoping the LLM "knows" the answer, you fetch the relevant information from your own documents first, hand it to the model, and then it answers. The model doesn't need to be smarter — it just needs to be better informed.

Built my first RAG pipeline today:
Load docs → Chunk → Embed → Store → Retrieve → Prompt → Generate

The biggest lesson? Bad chunking is the #1 reason RAG systems fail. Not bad models. Not bad prompts. Bad document splitting.

Day 1 done. Building an API Documentation Assistant from scratch. More coming soon.

#LangChain #RAG #AI #BuildInPublic #LearningInPublic #GenAI #NxtWave

---

## 🎨 3-Slide Carousel Prompts for ChatGPT Image Generation

Use these prompts in ChatGPT to generate each slide:

---

### SLIDE 1 (Hook / Title Slide)

**Prompt for ChatGPT:**
> Create a clean, modern LinkedIn carousel slide with a dark navy blue background (#0a192f). At the top, add a small subtle label that says "Day 1 — Learning in Public" in a muted teal color. In the center, display the main title in large bold white text: "3 Things I Learned About LLMs That Surprised Me". Below the title, add a thin horizontal teal accent line. At the bottom, add smaller text in gray: "NIAT LangChain Masterclass — Building a Production-Ready RAG Assistant". The design should feel minimal, premium, and developer-focused. No images, no icons, just clean typography. Aspect ratio 4:5.

---

### SLIDE 2 (Core Insights)

**Prompt for ChatGPT:**
> Create a clean LinkedIn carousel slide with a dark navy blue background (#0a192f). Display 3 numbered insights in white text with teal (#64ffda) numbers and accent highlights. The 3 points are:
>
> "1. LLMs don't remember anything — Every message is Day 1. Memory must be built separately."
>
> "2. LLMs can't do math — They predict the next word, not compute. That's why we give them calculator tools."
>
> "3. Bad chunking kills RAG — Not bad models, not bad prompts. How you split your documents determines answer quality."
>
> Use clean spacing between each point. Keep it minimal and readable with no images or icons. Add a thin teal line separator between each point. Aspect ratio 4:5.

---

### SLIDE 3 (What I Built + CTA)

**Prompt for ChatGPT:**
> Create a clean LinkedIn carousel slide with a dark navy blue background (#0a192f). At the top, show the text "What I Built Today" in teal (#64ffda). Below that, display a simple left-aligned flow diagram in white text showing these steps connected by arrows: "Load Docs → Chunk → Embed → Store → Retrieve → Prompt → Generate". Below the diagram, add the text in white: "A complete RAG pipeline that answers API questions using only my company's documentation." At the bottom, add a call-to-action in muted gray: "Follow along as I build a production-ready RAG assistant from scratch." Keep the design minimal, clean, and developer-focused. No images, just typography. Aspect ratio 4:5.

---

## 📋 Posting Checklist

- [ ] Copy the text post above as your LinkedIn caption
- [ ] Generate 3 carousel slides using ChatGPT image generation with the prompts above
- [ ] Combine the 3 images into a single PDF (for LinkedIn carousel upload)
- [ ] Upload the PDF as a "Document" on LinkedIn (this creates the carousel/swipe effect)
- [ ] Post timing: Best times are Tuesday-Thursday, 8-10 AM IST (but today/tomorrow works since it's fresh)
- [ ] Optionally tag: NIAT / NxtWave / Aakriti Agarwal
