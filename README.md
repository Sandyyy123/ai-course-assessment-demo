# AI Course Assessment Demo

An AI-powered conversational assessment engine that replaces multiple-choice exams with dynamic open-ended dialogue. Built with LangChain, OpenAI GPT-4o, ChromaDB, and FastAPI.

## Architecture

```
Course Content (PDF/slides)
        |
    ChromaDB (vector store)
        |
    Question Generator (LangChain RAG chain)
        |
    Dialogue Manager (multi-turn conversation)
        |
    Rubric Evaluator (separate LLM chain)
        |
    Feedback Report (score + gaps + recommendations)
        |
    LMS Integration (xAPI / SCORM / REST webhook)
```

## Features

- Open-ended question generation calibrated to course content
- Multi-turn conversation manager with context memory
- Dual-chain evaluation (question gen + answer eval are separate chains to prevent bias)
- Rubric-based scoring: accuracy, depth, reasoning, completeness (0-100)
- Personalized feedback with knowledge gap identification
- LMS score sync via xAPI statements or REST webhook
- Demo mode (no API key required) with realistic mock responses

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
uvicorn main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/start_session` | Start assessment for a learner + module |
| POST | `/respond` | Submit learner answer, get next question |
| GET | `/result/{session_id}` | Get final score + feedback report |
| POST | `/ingest` | Upload course content to vector store |

## Demo Mode

Run without an OpenAI API key - the system returns realistic mock responses demonstrating the full conversation flow.

```bash
DEMO_MODE=true uvicorn main:app --reload
```

## LMS Integration

The `/result` endpoint returns an xAPI statement ready to POST to your LRS:

```json
{
  "actor": {"name": "learner_id"},
  "verb": {"id": "http://adlnet.gov/expapi/verbs/scored"},
  "result": {"score": {"scaled": 0.87}, "completion": true},
  "object": {"id": "course/module-3"}
}
```
