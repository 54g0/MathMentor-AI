<div align="center">

# 🧮 MathMentor AI
An agentic, retrieval‑augmented Math Tutor with a feedback refinement loop and self‑growing vector knowledge base.

</div>

## 🌟 What Is This?
MathMentor AI is a full‑stack learning assistant that:
- Accepts only math‑related queries (non‑math inputs are politely declined up front).
- Uses tool-augmented reasoning (retrieval + optional web search) before solving.
- Shows a step‑by‑step solution, not just a final number.
- Lets you give natural feedback (e.g., “Explain with a diagram idea” / “Show factoring steps”).
- Generates an improved answer incorporating your feedback.
- Stores the original Question + Answer pair into a local FAISS vector database so future retrieval becomes richer over time.

This is more than a plain LLM demo—it’s an iterative teaching loop with memory growth.

## 🔑 Unique Qualities
| Feature | Why It Matters |
|---------|----------------|
| Math-only gate | Prevents model drift into irrelevant chit‑chat. |
| Mandatory retrieval-first intent | Encourages grounded reasoning before solution. |
| Feedback refinement agent | Lets the learner steer style, depth, granularity. |
| Self-growing vector memory | Each solved problem enriches future context. |
| Modular MCP tools | Easily extend with new specialized tools. |
| Clean separation (UI / API / Agent) | Swap any layer without refactoring the others. |

## 📂 Key Files
| Path | Purpose |
|------|---------|
| `backend/agent.py` | Core MathTutorAgent + FeedbackAgent. Enforces math gating & tool-first reasoning prompt. |
| `backend/mcp_server.py` | Exposes `retrieve_data` and `web_search` MCP tools. |
| `backend/api_server.py` | FastAPI endpoints: `/health`, `/ask`, `/feedback` (and auto Q/A vector DB ingestion). |
| `backend/KB_setup.py` | (Run once) Builds or initializes the vector database. |
| `backend/benchmark.py` | Simple accuracy benchmarking on JEE-style MCQs. |
| `backend/vdb_updater.py` or `vector_db/updater.py` | Helper that appends new Q/A pairs to FAISS index. |
| `frontend/app.py` | Streamlit chat UI with feedback form and improved answer display. |

## 🛠️ Prerequisites
- Python 3.11+ recommended
- GPU not required (but helpful for large embedding batches)
- An API key for your selected model provider (e.g., Groq) exported via environment variables

## 🚀 Quick Start (Local)
Clone the public template repo (example reference given):
```bash
git clone https://github.com/54g0/MathMentor-AI
cd MathMentor-AI
```

Create & activate a virtual environment and install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Build / Initialize the Knowledge Base
Run the setup script (adjust path if you want a custom storage location):
```bash
python backend/KB_setup.py --out vector_db/vector_store
```

### 2. Start the MCP Tool Server
```bash
python backend/mcp_server.py
```
This provides the `retrieve_data` (vector similarity) and `web_search` tool endpoints used by the agent.

### 3. Start the API Server
```bash
uvicorn backend.api_server:app --host 0.0.0.0 --port 8010
```

### 4. Launch the Frontend UI
```bash
streamlit run frontend/app.py
```
Open the printed local URL (usually http://localhost:8501).

You’re now ready to interact.

## 🧪 Interaction Workflow (What Happens Internally)
1. Input Gate: The agent first classifies your input—if it is not math-related it rejects politely (no wasted tokens).
2. Retrieval Phase: Runs `retrieve_data` against FAISS vector DB for prior similar Q/A context.
3. Optional Web Search: If retrieval seems insufficient (based on instructions) it may call `web_search`.
4. Chain-of-Thought Style Reasoning: Produces a structured, step-by-step solution.
5. Feedback Loop: UI displays a “Give Feedback” button. You can request stylistic or structural changes (e.g., “Show factorization steps first” or “Use a comparison table”).
6. Refinement: A lightweight feedback agent rewrites the answer according to your guidance.
7. Memory Growth: The original (question, answer) pair is appended to the vector store for future retrieval enrichment.

## 🗃️ Vector Store Behavior
- Format stored: Plain text blocks in the form: `Q: ...\nA: ...`
- Engine: FAISS + sentence-transformer embedding (`all-MiniLM-L6-v2`).
- Persistence: Updated on every successful `/ask` response.
- Extension Ideas: Add metadata (timestamp, difficulty), deduplicate by hash, schedule periodic compaction.

## 🔁 Feedback Examples
| Feedback You Give | What Happens |
|-------------------|--------------|
| “Show each algebra step explicitly.” | Regenerates answer with granular transformations. |
| “Summarize only final formula.” | Produces a concise final-form answer. |
| “Explain like I’m 12.” | Shifts tone + simplifies vocabulary. |
| “Use bullet points for reasoning.” | Reformats steps into list structure. |

## 📊 Benchmarking (Optional)
Run a small accuracy check on JEE-style MCQs:
```bash
python backend/benchmark.py --max 30
```
Outputs an overall accuracy ratio (predicted option vs correct option). The benchmark is intentionally minimal.

## ⚙️ Environment Variables
| Name | Purpose | Default |
|------|---------|---------|
| MODEL_PROVIDER | LLM backend provider | groq |
| MODEL_NAME | Model identifier | openai/gpt-oss-120b |
| DEBUG | Extra logging (agent / vector updates) | false |
| API_URL (frontend) | Backend base URL | http://localhost:8010 |

Set via shell export or an `.env` file.

## 🤝 Contributing
Fork, branch, and open a PR. For larger ideas, open an issue first to discuss approach (especially for new tool interfaces or memory schemas).

## 📄 License
Educational / internal prototype. Add an explicit license file if redistributing.

---
Made to explore how iterative, tool-grounded tutoring + user feedback can build an evolving math knowledge base.

Happy learning! 🧠
