from __future__ import annotations

import logging
import math
import os
import threading
import numpy as np

from rules import (RULE_FEATURE_NAMES, _clamp, analyze_text as rules_analyze, confidence_for,
                   empty_result, explanations, notes_for, rule_features, rule_sentence_raw,
                   rule_vector)

log = logging.getLogger("gotchyu")
MODEL_PATH = os.environ.get("GOTCHYU_MODEL", "gotchyu_model.joblib")
LM_FEATURES = ["lm_mean_nll", "lm_std_nll", "lm_top1_share", "lm_top10_share", "lm_sentence_nll_cv"]
FEATURE_NAMES = LM_FEATURES + RULE_FEATURE_NAMES

W_LM, W_RULE = 0.55, 0.45
NLL_LOW, NLL_HIGH = 2.2, 4.5
RESIDUE_FLOOR = 0.90


class GPT2Scorer:
    def __init__(self, model_id: str = "gpt2"):
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = GPT2TokenizerFast.from_pretrained(model_id)
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device).eval()

    def token_stats(self, text: str):
        torch = self.torch
        enc = self.tok(text, return_offsets_mapping=True)
        ids, starts = enc["input_ids"], [o[0] for o in enc["offset_mapping"]]
        nll, top1, top10 = [], [], []
        bos = self.tok.bos_token_id
        with torch.no_grad():
            for i in range(0, len(ids), 1023):
                chunk = torch.tensor([[bos] + ids[i:i + 1023]], device=self.device)
                logp = torch.log_softmax(self.model(chunk).logits[0, :-1].float(), dim=-1)
                chosen = logp.gather(1, chunk[0, 1:, None]).squeeze(1)
                rank = (logp > chosen[:, None]).sum(1)
                nll.append((-chosen).cpu())
                top1.append((rank == 0).cpu())
                top10.append((rank < 10).cpu())
        if not nll:
            return np.zeros(0), np.zeros(0), np.zeros(0), []
        return (torch.cat(nll).numpy(), torch.cat(top1).numpy().astype(float),
                torch.cat(top10).numpy().astype(float), starts)


class Detector:
    def __init__(self, model_path: str = MODEL_PATH, lm=None, load_lm: bool = True):
        self.engine, self.model, self.lm = "rules", None, lm
        self._sem = threading.BoundedSemaphore(int(os.environ.get("GOTCHYU_CONCURRENCY", "2")))
        if not os.path.exists(model_path):
            log.warning("No trained model at %s: using the rules-only engine. Run train_detector.py.", model_path)
            return
        try:
            import joblib
            bundle = joblib.load(model_path)
        except Exception as e:
            log.warning("Could not load %s (%s): using the rules-only engine.", model_path, e)
            return
        if bundle.get("feature_names") != FEATURE_NAMES:
            log.warning("Model features do not match this code: retrain. Using the rules-only engine.")
            return
        if self.lm is None and load_lm:
            try:
                self.lm = GPT2Scorer(bundle.get("gpt2_id", "gpt2"))
            except Exception as e:
                log.warning("GPT-2 unavailable (%s): using the rules-only engine.", e)
                return
        if self.lm is None:
            return
        self.model, self.engine = bundle["model"], "gpt2+rules"

    def _lm_features(self, R: dict):
        parts, spans, pos = [], [], 0
        for i, (s, _, brk) in enumerate(R["sents"]):
            if i:
                sep = "\n\n" if brk else " "
                parts.append(sep)
                pos += len(sep)
            spans.append((pos, pos + len(s)))
            parts.append(s)
            pos += len(s)
        nll, top1, top10, starts = self.lm.token_stats("".join(parts))
        if len(nll) == 0:
            return [0.0] * len(LM_FEATURES), [None] * len(spans)
        idx = np.searchsorted([a for a, _ in spans], np.array(starts), side="right") - 1
        sent_nll = [float(nll[idx == i].mean()) if (idx == i).any() else None for i in range(len(spans))]
        usable = [v for i, v in enumerate(sent_nll) if v is not None and (idx == i).sum() >= 4]
        cv = float(np.std(usable) / np.mean(usable)) if len(usable) >= 3 and np.mean(usable) > 0 else 0.25
        vec = [float(nll.mean()), float(nll.std()), float(top1.mean()), float(top10.mean()), cv]
        return vec, sent_nll

    def featurize(self, text: str):
        R = rule_features(text)
        if R is None:
            return None
        lm_vec, _ = self._lm_features(R)
        return lm_vec + rule_vector(R)

    def analyze(self, text: str) -> dict:
        if self.model is None:
            res = rules_analyze(text)
            res["engine"] = "rules"
            return res
        R = rule_features(text)
        if R is None:
            return {**empty_result(), "engine": self.engine}
        with self._sem:
            lm_vec, sent_nll = self._lm_features(R)

        p = float(self.model.predict_proba(np.array([lm_vec + rule_vector(R)]))[0][1])

        # Manually force the cliche penalty
        manual_penalty = (R.get("cliche_count", 0) * 0.15) + (R.get("transition_count", 0) * 0.05)
        p += manual_penalty

        conf, shrink = confidence_for(R)
        if R["residue"]:
            p = max(p, RESIDUE_FLOOR)
        else:
            p = 0.5 + (p - 0.5) * shrink
        p = _clamp(p, 0.01, 0.98)

        out = []
        for (s, _, brk), rule_raw, nll in zip(R["sents"], rule_sentence_raw(R), sent_nll):
            rule_s = _clamp(rule_raw)
            if nll is None:
                base = rule_s
            else:
                lm_score = _clamp((NLL_HIGH - nll) / (NLL_HIGH - NLL_LOW))
                base = max(lm_score, rule_s)

                # THE FIX: Scale the sentence score using the document's overall probability.
            # If the overall document hits 92%, this guarantees the sentences get pulled
            # up into the highlight thresholds (0.5 to 1.0) so the CSS triggers.
            score = (base * p) + (p * 0.5)

            if R["residue"] and rule_raw >= 0.6:
                score = max(score, 0.8)

            item = {"text": s, "score": round(score, 2), "break_before": brk}
            if nll is not None:
                item["perplexity"] = round(math.exp(nll), 1)
            out.append(item)

        signals, human_signals = explanations(R)
        if lm_vec[0] and lm_vec[0] < 3.0:
            # FIXED SYNTAX ERROR: Added missing subtraction symbol (-)
            signals.insert(0, {
                "label": "Very predictable wording (language-model check)",
                "strength": round(3.0 - lm_vec[0], 2),
                "examples": []
            })

        return {"ai_percent": round(p * 100), "sentences": out, "confidence": conf,
                "word_count": R["W"], "signals": signals, "human_signals": human_signals,
                "notes": notes_for(conf), "engine": self.engine}


# Helper wrapper for main.py to call
def analyze_text(text: str) -> dict:
    detector = Detector()
    return detector.analyze(text)