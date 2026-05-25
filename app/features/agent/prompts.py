SELECT_MASTER_PROMPT = """Given a handwritten sketch analysis and candidate master drawings, pick the best matching master.

Sketch analysis:
{analysis}

Candidates:
{candidates}

Return ONLY JSON:
{{"master_key": "Category/basename", "confidence": 0.0-1.0, "reasoning": "why this master matches"}}
"""
