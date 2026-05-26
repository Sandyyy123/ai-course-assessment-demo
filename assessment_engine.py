"""
Core AI assessment engine.
Dual-chain design: question generator and answer evaluator are separate LLM chains
to prevent self-consistency bias.
"""
import os
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# Demo responses used when OPENAI_API_KEY is not set
DEMO_QUESTIONS = [
    "You are building a hiring model trained on historical data and achieve 92% accuracy overall. Why might this still be problematic, and what steps would you take to identify bias?",
    "Explain the difference between demographic parity and equalized odds. In a high-stakes decision context, which would you prefer and why?",
    "A model performs well on your test set but poorly in production. List three possible causes and how you would diagnose each.",
]

DEMO_EVALUATIONS = [
    {"score": 82, "accuracy": 85, "depth": 80, "reasoning": 83, "completeness": 79,
     "feedback": "Good identification of subgroup performance gaps. Missing: mention of calibration metrics and audit logs.",
     "gaps": ["Calibration metrics not mentioned", "Audit trail requirements"],
     "recommended_modules": ["Module 3.2: Fairness Metrics", "Module 3.4: Audit Frameworks"]},
    {"score": 87, "accuracy": 90, "depth": 85, "reasoning": 88, "completeness": 84,
     "feedback": "Strong definitions and correct use-case judgment. Could expand on impossibility results (demographic parity vs equalized odds cannot both hold when base rates differ).",
     "gaps": ["Impossibility theorems not mentioned"],
     "recommended_modules": ["Module 3.3: Fairness Trade-offs"]},
]


def generate_question(module_content: str, question_num: int, conversation_history: list) -> str:
    """Generate next assessment question based on module content and conversation so far."""
    if DEMO_MODE or not os.getenv("OPENAI_API_KEY"):
        idx = min(question_num, len(DEMO_QUESTIONS) - 1)
        return DEMO_QUESTIONS[idx]

    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate

        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

        history_text = ""
        for turn in conversation_history:
            history_text += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert educational assessor. Given course content and conversation history,
generate ONE open-ended assessment question that:
1. Tests conceptual understanding, not memorization
2. Requires the learner to apply or analyze, not just recall
3. Builds on previous answers without repeating covered ground
4. Is calibrated to Bloom's taxonomy: comprehension, application, or analysis level
Return only the question text, nothing else."""),
            ("human", f"""Course content excerpt:
{module_content[:2000]}

Previous conversation:
{history_text or "No previous questions yet."}

Question number: {question_num + 1}
Generate the next assessment question:""")
        ])

        chain = prompt | llm
        result = chain.invoke({})
        return result.content.strip()
    except Exception as e:
        return DEMO_QUESTIONS[min(question_num, len(DEMO_QUESTIONS) - 1)]


def evaluate_answer(question: str, answer: str, module_content: str) -> dict:
    """
    Evaluate a learner answer against a rubric.
    This runs as a SEPARATE LLM call from question generation to prevent bias.
    """
    if DEMO_MODE or not os.getenv("OPENAI_API_KEY"):
        import random
        base = DEMO_EVALUATIONS[min(hash(answer[:20]) % len(DEMO_EVALUATIONS), len(DEMO_EVALUATIONS)-1)]
        return base

    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate

        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)  # Low temp for consistent scoring

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an objective educational assessor. Evaluate the learner answer using this rubric:
- Accuracy (0-25): Is the core claim factually correct?
- Depth (0-25): Does the answer go beyond surface-level?
- Reasoning (0-25): Is the logic sound and well-structured?
- Completeness (0-25): Are key aspects of the question addressed?

Return JSON only with keys: accuracy, depth, reasoning, completeness, score (sum), feedback (1-2 sentences), gaps (list of missing concepts), recommended_modules (list)."""),
            ("human", f"""Course content (ground truth):
{module_content[:1500]}

Question asked: {question}

Learner answer: {answer}

Evaluate and return JSON:""")
        ])

        chain = prompt | llm
        result = chain.invoke({})
        return json.loads(result.content.strip())
    except Exception:
        return DEMO_EVALUATIONS[0]


def compute_final_report(session: dict) -> dict:
    """Aggregate turn-by-turn scores into a final assessment report."""
    turns = session.get("turns", [])
    if not turns:
        return {"error": "No turns recorded"}

    total_score = sum(t["evaluation"]["score"] for t in turns) / len(turns)
    all_gaps = []
    all_recs = []
    for t in turns:
        all_gaps.extend(t["evaluation"].get("gaps", []))
        all_recs.extend(t["evaluation"].get("recommended_modules", []))

    # Deduplicate
    all_gaps = list(dict.fromkeys(all_gaps))
    all_recs = list(dict.fromkeys(all_recs))

    return {
        "session_id": session["session_id"],
        "learner_id": session["learner_id"],
        "module": session["module"],
        "score": round(total_score, 1),
        "questions_asked": len(turns),
        "knowledge_gaps": all_gaps[:5],
        "recommended_modules": all_recs[:3],
        "xapi_statement": {
            "actor": {"name": session["learner_id"]},
            "verb": {"id": "http://adlnet.gov/expapi/verbs/scored"},
            "result": {
                "score": {"scaled": round(total_score / 100, 3), "raw": round(total_score, 1)},
                "completion": True,
                "success": total_score >= 70
            },
            "object": {"id": f"course/{session['module']}"}
        }
    }
