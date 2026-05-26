"""
AI Course Assessment API
FastAPI entry point - conversational assessment engine replacing MCQ exams.
"""
import uuid
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from assessment_engine import generate_question, evaluate_answer, compute_final_report

load_dotenv()

app = FastAPI(
    title="AI Course Assessment Engine",
    description="Replaces MCQ exams with conversational AI assessment + LMS score sync",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory session store (replace with PostgreSQL for production)
SESSIONS = {}

# Mock course content (replace with ChromaDB retrieval in production)
DEMO_CONTENT = """
Data Ethics and Fairness in Machine Learning:
- Algorithmic bias arises when training data reflects historical inequities
- Aggregate accuracy metrics can mask disparate performance across subgroups
- Fairness metrics: demographic parity (equal positive rates), equalized odds (equal TPR/FPR per group)
- Calibration: model confidence should match actual outcome rates
- Mitigation: pre-processing (data resampling), in-processing (fairness constraints), post-processing (threshold adjustment)
- Audit requirements: disaggregated confusion matrices, protected class analysis, documentation
"""


class StartRequest(BaseModel):
    learner_id: str
    module: str
    num_questions: int = 3


class RespondRequest(BaseModel):
    session_id: str
    answer: str


@app.post("/start_session")
def start_session(req: StartRequest):
    """Start a new assessment session for a learner on a specific module."""
    session_id = str(uuid.uuid4())[:8]

    # In production: retrieve module content from ChromaDB
    module_content = DEMO_CONTENT

    first_question = generate_question(module_content, 0, [])

    session = {
        "session_id": session_id,
        "learner_id": req.learner_id,
        "module": req.module,
        "num_questions": req.num_questions,
        "module_content": module_content,
        "turns": [],
        "current_question": first_question,
        "status": "active"
    }
    SESSIONS[session_id] = session

    return {
        "session_id": session_id,
        "question": first_question,
        "question_num": 1,
        "total_questions": req.num_questions
    }


@app.post("/respond")
def respond(req: RespondRequest):
    """Submit a learner answer. Returns evaluation + next question (or completion)."""
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] != "active":
        raise HTTPException(status_code=400, detail="Session already completed")

    current_q = session["current_question"]

    # Evaluate this answer
    evaluation = evaluate_answer(current_q, req.answer, session["module_content"])

    session["turns"].append({
        "question": current_q,
        "answer": req.answer,
        "evaluation": evaluation
    })

    question_num = len(session["turns"])

    if question_num >= session["num_questions"]:
        session["status"] = "complete"
        report = compute_final_report(session)
        return {
            "status": "complete",
            "turn_score": evaluation.get("score"),
            "turn_feedback": evaluation.get("feedback"),
            "final_report": report
        }

    # Generate next question
    next_question = generate_question(
        session["module_content"],
        question_num,
        session["turns"]
    )
    session["current_question"] = next_question

    return {
        "status": "active",
        "turn_score": evaluation.get("score"),
        "turn_feedback": evaluation.get("feedback"),
        "question": next_question,
        "question_num": question_num + 1,
        "total_questions": session["num_questions"]
    }


@app.get("/result/{session_id}")
def get_result(session_id: str):
    """Get the final assessment report + xAPI statement for LMS sync."""
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] != "complete":
        raise HTTPException(status_code=400, detail="Session still active")
    return compute_final_report(session)


@app.get("/health")
def health():
    return {"status": "ok", "demo_mode": os.getenv("DEMO_MODE", "false")}
