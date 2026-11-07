import re

RULE_FEATURE_NAMES = ["ai_cliches", "transitions"]

# Common algorithmic footprint words and phrases
CLICHES = r"\b(tapestry|symphony|testament|beacon|realm|delve|seamlessly|meticulously|opulent|fostering|multifaceted|nuanced|intricate)\b"
TRANSITIONS = r"\b(stands as|serves as|ultimately|furthermore|moreover|consequently|it is crucial to|in conclusion|to summarize)\b"


def _clamp(val, min_val=0.0, max_val=1.0):
    return max(min_val, min(val, max_val))


def empty_result():
    return {"ai_percent": 0, "sentences": [], "confidence": "Low", "word_count": 0, "signals": [], "human_signals": [],
            "notes": []}


def rule_features(text):
    if not text.strip():
        return None

    # Split text into rough sentences for highlighting
    raw_sents = [s.strip() + "." for s in text.replace("\n", " ").split('.') if s.strip()]
    sents = [(s, None, False) for s in raw_sents]
    words = len(text.split())

    return {
        "text": text,
        "sents": sents,
        "W": words,
        "residue": bool(re.search(r"as an ai language model|as an ai", text, re.IGNORECASE)),
        "cliche_count": len(re.findall(CLICHES, text, re.IGNORECASE)),
        "transition_count": len(re.findall(TRANSITIONS, text, re.IGNORECASE))
    }


def rule_vector(R):
    # Convert raw counts to frequency per 100 words so the ML model can scale it
    w_scale = max(1, R["W"] / 100)
    return [R["cliche_count"] / w_scale, R["transition_count"] / w_scale]


def rule_sentence_raw(R):
    # Scores individual sentences to trigger the UI highlighter
    out = []
    for s, _, _ in R["sents"]:
        c = len(re.findall(CLICHES, s, re.IGNORECASE))
        t = len(re.findall(TRANSITIONS, s, re.IGNORECASE))
        out.append(_clamp((c + t) * 0.4))
    return out


def explanations(R):
    signals = []
    if R["cliche_count"] > 0:
        signals.append({"label": "Contains common AI vocabulary", "strength": 0.8, "examples": []})
    return signals, []


def notes_for(conf):
    return []


def confidence_for(R):
    W = R["W"]
    shrink = _clamp((W - 40) / 260, 0.1, 1.0)   # 40 words -> 0.1, 300+ words -> 1.0
    conf = "Low" if W < 100 else "Medium" if W < 250 else "High"
    return conf, shrink


def analyze_text(text):
    return empty_result()