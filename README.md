# LLM Jailbreak Detection System

A production-grade multi-layer adversarial prompt detection system for large language models. It combines deterministic rule matching, fine-tuned transformer classification, and semantic similarity search to identify and block jailbreak attempts, prompt injections, and other adversarial inputs.


---

## Results

| Metric | Score |
|---|---|
| Binary F1 (attack vs benign) | **1.000** |
| False Positive Rate | **0.000** |
| ML Classifier F1 (standalone) | 0.884 |
| Weighted F1 (7-class) | 0.722 |

**Layer Ablation**

| Configuration | Binary F1 | Latency |
|---|---|---|
| Rule engine only | 0.456 | 2ms |
| Rule + ML classifier | 0.989 | 36ms |
| **Full system (all 3 layers)** | **1.000** | 253ms |

---

## Architecture

```
Input → Preprocessor (decode + normalize)
           │
    ┌──────┼──────────┐
    ▼      ▼          ▼
 Rules   DistilBERT  FAISS
 ×0.40   ×0.35       ×0.25
    └──────┼──────────┘
           ▼
      Risk Engine
   BLOCK / MONITOR / ALLOW
           │
    ┌──────┴──────┐
  FastAPI      Streamlit
```

**Layer 1 — Rule Engine:** 33 regex patterns, severity 1–5, zero ML overhead.

**Layer 2 — ML Classifier:** Fine-tuned `distilbert-base-uncased`, 7-class, F1=0.884.

**Layer 3 — Embedding Matcher:** `all-mpnet-base-v2` + FAISS, catches novel paraphrased attacks.

---

## Attack Categories

`direct_jailbreak` · `prompt_injection` · `roleplay_hijack` · `encoding_attack` · `manyshot` · `indirect_injection`

---

## Quickstart

```bash
# Setup
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Train
python scripts/download_datasets.py
python scripts/train_classifier.py
python scripts/build_index.py

# Run API (Terminal 1)
uvicorn src.jailbreak_detection.api.main:app --port 8000 --reload

# Run Dashboard (Terminal 2)
streamlit run src/jailbreak_detection/dashboard/app.py
```

- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

### Docker

```bash
docker compose up
```

---

## API

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "ignore all previous instructions"}'
```

```json
{
  "decision": "BLOCK",
  "risk_score": 0.95,
  "attack_category": "direct_jailbreak",
  "severity": 5,
  "is_attack": true
}
```

---

## Tests

226 tests across 7 files. Run all:

```bash
python tests/test_preprocessing.py       # 14 tests
python tests/test_rule_engine.py         # 26 tests
python tests/test_classifier.py          # 37 tests
python tests/test_embedding_matcher.py   # 34 tests
python tests/test_aggregator.py          # 48 tests
python tests/test_api.py                 # 40 tests
python tests/test_evaluation.py          # 27 tests
```

---

## Tech Stack

PyTorch · HuggingFace Transformers · FAISS · FastAPI · Streamlit · Docker

---

## Note on Scope

This system detects **jailbreak mechanics**, including instruction overrides, encoding attacks, and roleplay framing. Direct harmful questions without a jailbreak structure score low by design and are handled by the downstream LLM's own safety layer.