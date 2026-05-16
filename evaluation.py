from pathlib import Path

import pandas as pd


def load_evaluation_questions(path="data/evaluation_questions.csv"):
    eval_path = Path(path)

    if not eval_path.exists():
        return pd.DataFrame(columns=["question", "expected_terms", "notes"])

    return pd.read_csv(eval_path)


def score_answer_quality(question, answer, evidence_items, eval_df):
    """Score answer quality against the closest evaluation row when available."""
    answer_text = str(answer).lower()
    evidence_count = len(evidence_items)

    if eval_df.empty:
        return min(100, 45 + evidence_count * 10)

    question_lower = question.lower()
    eval_df = eval_df.copy()
    eval_df["overlap"] = eval_df["question"].str.lower().apply(
        lambda candidate: len(set(question_lower.split()) & set(candidate.split()))
    )
    best = eval_df.sort_values("overlap", ascending=False).iloc[0]

    if best["overlap"] == 0:
        return min(100, 45 + evidence_count * 10)

    expected_terms = [
        term.strip().lower()
        for term in str(best["expected_terms"]).split(";")
        if term.strip()
    ]

    if not expected_terms:
        return min(100, 45 + evidence_count * 10)

    matched_terms = sum(1 for term in expected_terms if term in answer_text)
    keyword_score = (matched_terms / len(expected_terms)) * 70
    evidence_score = min(30, evidence_count * 6)

    return round(keyword_score + evidence_score)
