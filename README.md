# 🔊 PulseAgent

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green.svg)](https://langchain-ai.github.io/langgraph/)

A multi-agent AI system that collects product reviews from the web, classifies and clusters them by issue type, detects churn signals, generates draft responses for your support team, and produces a prioritized product roadmap — all in one pipeline.

---

## How it works

```
Reviews (Reddit · App Store · Fixtures)
        ↓
  ClassifierAgent   — categorizes each review (bug, feature request, churn signal…)
        ↓
  ScorerAgent       — scores urgency, groups into clusters
        ↓
  RAGAgent          — checks product docs: is this issue already fixed?
        ↓
  ResponseAgent     — writes draft replies for high-urgency reviews
        ↓
  RoadmapAgent      — generates P0–P3 backlog items with user stories
        ↓
  TrendAgent        — detects rising/falling issues over time
        ↓
  CLI dashboard / FastAPI / Streamlit UI
```

**LLMs used:** Groq (fast classification) · Gemini (reasoning, responses, roadmap) — both free tier.

---

## Quickstart

**1. Install**
```bash
git clone https://github.com/0xkerem/pulseagent.git
cd pulseagent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add API keys**
```bash
cp .env.example .env
# Edit .env:
# GROQ_API_KEY=...    → https://console.groq.com  (free)
# GEMINI_API_KEY=...  → https://aistudio.google.com  (free)
```

**3. Run**
```bash
# CLI — uses built-in Notion demo data
python run.py --product notion --limit 30

# Dashboard
streamlit run dashboard/app.py

# API
uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

**4. Docker**
```bash
docker-compose up --build
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## Project structure

```
pulseagent/
├── agents/          # scraper, classifier, scorer, rag, response, roadmap, trend
├── graph/           # LangGraph pipeline wiring all agents
├── api/             # FastAPI REST + WebSocket
├── dashboard/       # Streamlit UI
├── data/
│   ├── fixtures/    # Demo review data (Notion, default)
│   └── docs/        # Product docs for RAG
├── infra/terraform/ # GCP Cloud Run deployment
├── tests/           # pytest suite
├── config.py        # Settings via .env
├── models.py        # Shared Pydantic models
├── llm_factory.py   # Groq / Gemini router
└── run.py           # CLI entry point
```

---

## Stack

| | |
|---|---|
| Agent framework | LangGraph |
| LLM — fast | Groq (llama-3.1-70b) |
| LLM — reasoning | Gemini (gemini-1.5-flash) |
| Vector DB | ChromaDB |
| API | FastAPI async |
| Dashboard | Streamlit + Plotly |
| Deploy | Docker · GCP Cloud Run · Terraform |
| Tests | pytest · pytest-asyncio |

---

## Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

---
