# RBAC Access Control System — Role-Based AI Document Assistant

A Retrieval-Augmented Generation (RAG) system for multi-role enterprise environments, built to solve a real operational problem: siloed, department-specific data that slows down decision-making. Users authenticate with a role, ask questions in plain English, and the system automatically routes each query to the right engine — structured SQL or unstructured document search — while enforcing strict role-based access at every step.

Originally prototyped with OpenAI GPT-4o, the stack has been rebuilt end-to-end on free-tier services (Groq, HuggingFace, Cohere) so the whole system runs without a paid LLM API key.

## Business Problem

FinSolve Technologies faced operational inefficiencies caused by communication delays and fragmented data across Finance, Marketing, HR, and Executive leadership. Departmental isolation slowed decision-making and project execution. This system delivers secure, on-demand, department-specific insights while enforcing strict access controls to protect data confidentiality.

## What It Does

1. **Authenticate** — users log in with role-based credentials (Finance, HR, Marketing, Engineering, C-Level, General).
2. **Ask a question** in plain English.
3. **Automatic query routing** — an LLM classifier decides whether the question needs:
   - **Structured data lookup (SQL)** — e.g. *"How many employees have a performance rating of 5?"* → generates and safely executes SQL against a DuckDB warehouse.
   - **Document search (RAG)** — e.g. *"Summarize our leave policy"* → semantic search over embedded markdown/CSV documents.
4. **Role-scoped retrieval** — every retrieval is filtered by the user's role before it ever reaches the LLM. A Marketing employee's query can never surface Finance or HR documents, regardless of how the question is phrased.
5. **Fallback handling** — if the SQL path fails or returns nothing useful, the query automatically reroutes to the RAG pipeline with a rephrased prompt, so the user never hits a hard error.
6. **Reranking** — Cohere Rerank re-scores retrieved chunks for semantic relevance before they reach the LLM, improving answer trustworthiness.
7. **Answer generation** — retrieved context + the question are passed to an LLM to produce a grounded, cited answer.
8. **Evaluation** — an automated QA pipeline generates question/answer pairs from the source documents and scores model output on faithfulness, relevance, and conciseness.

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI, SQLite (users/roles), DuckDB (structured data) |
| **RAG** | LangChain 1.x, ChromaDB (vector store), HuggingFace sentence-transformers (embeddings) |
| **LLM** | Groq (Llama 3.3 70B) — swapped from OpenAI GPT-4o to run on a free tier |
| **Reranking** | Cohere Rerank |
| **Frontend** | Streamlit |
| **Auth** | HTTP Basic Auth + bcrypt password hashing |
| **Testing** | Pytest, Playwright (UI automation) |

## Architecture

```
User (Streamlit UI)
        │
        ▼
   FastAPI backend
        │
        ▼
 Query Classifier Agent
   ┌────┴────┐
   ▼         ▼
SQL Agent   RAG Agent
(Groq LLM   (HuggingFace embeddings
→ DuckDB)   → ChromaDB → Cohere Rerank
   │         → Groq LLM)
   │         │
   └────┬────┘
        ▼
   Fallback logic
   (SQL fails → reroute to RAG)
        │
        ▼
  Grounded response to user
```

- **Dual-mode query handling:** structured queries hit DuckDB directly; unstructured queries go through vector search + reranking + generation.
- **Fallback design:** the user always gets a meaningful answer, even if the first path fails.
- **Role-based filtering:** applied at the retrieval layer, not just the UI, so access control can't be bypassed by query phrasing.

## Why DuckDB for Structured Queries

- In-process SQL engine — runs embedded in Python, no separate server.
- Zero setup — no configuration required for file-based structured queries.
- Lightweight and fast — handles large CSV files efficiently in memory.
- Native Pandas + SQL support — easy to move between dataframes and SQL.
- Isolated execution — each user session can be sandboxed.

## Roles and Permissions

| Role | Permissions |
|---|---|
| Finance Team | Financial reports, marketing expenses, equipment costs, reimbursements |
| Marketing Team | Campaign performance data, customer feedback, sales metrics |
| HR Team | Employee data, attendance records, payroll, performance reviews |
| Engineering Dept. | Technical architecture, development processes, operational guidelines |
| C-Level Executives | Full access to all company data |
| Employee Level | General company info — policies, events, FAQs |

## Sample Queries

- *"Give me a summary about system architecture"* — Engineering
- *"Give me the details of employees in the Data department whose performance rating is 5"* — HR
- *"What percentage of the Vendor Services expense was allocated to marketing-related activities?"* — Finance
- *"What is the Return on Investment (ROI) for FinSolve Technologies?"* — Finance
- *"Give me details about leave policies."* — General
- *"What was the percentage increase in net income in 2024?"* — Finance

## Project Structure

```
├── app
│   ├── main.py                    # FastAPI backend
│   ├── rag_evaluator/             # RAG evaluation & results
│   ├── rag_utils/                 # RAG & SQL agents, query classifier
│   └── ui.py                      # Streamlit frontend
├── assets/style.css
├── requirements.txt
├── resources/                     # FinSolve sample data
├── static/
│   ├── data/structured_queries.duckdb
│   └── uploads/                   # role-tagged uploaded docs
├── tests/                         # Pytest + Playwright suites
└── videos/                        # Playwright demo recordings
```

## Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/ThejaswiniSunil/rbac-access-control-system.git
cd rbac-access-control-system
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API keys**

Create a `.env` file (or `secret_key.py`) with:
```
GROQ_API_KEY=your_groq_key
COHERE_API_KEY=your_cohere_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

**4. Run the application**

Terminal 1 — start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

Terminal 2 — start the Streamlit UI:
```bash
streamlit run app/ui.py
```

Then open: `http://localhost:8501`

**5. Sample users**

| Username | Password | Role |
|---|---|---|
| Tony | password123 | Engineering |
| Bruce | securepass | Marketing |
| Sam | financepass | Finance |
| Natasha | hrpass123 | HR |
| Nolan | nolan123 | General |

**6. Run tests**

Backend:
```bash
pytest tests/test_chatbot.py --html=report.html
```

UI (with both servers running):
```bash
pytest tests/test_ui.py --headed
```

**7. Run RAG evaluation**
```bash
python app/rag_evaluator/evaluator.py
```
Outputs:
- `qa_pairs.csv` — synthetic QA pairs
- `evaluation_results.csv` — model predictions with evaluation scores
- `final_eval_with_roles.csv` — final predictions broken down by role

## Future Enhancements

- Admin analytics dashboard (query types, usage patterns)
- Hybrid table + text retrieval (RAG with tabular fusion)
- Caching of repeated SQL queries

## Conclusion

This project demonstrates a flexible, intelligent retrieval pipeline that dynamically routes user queries to either structured (SQL) or unstructured (RAG) engines, with role-based access enforced throughout, reranking for precision, and automated evaluation at every layer — built entirely on free-tier infrastructure.
