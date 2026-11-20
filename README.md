# 🛡️ LLM Sentinel

> **Adversarial Security Evaluation & Vulnerability Auditing Framework for LLM and RAG Applications**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![OWASP LLM Top 10](https://img.shields.io/badge/OWASP-LLM%20Top%2010-red.svg)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
[![Tests](https://img.shields.io/badge/tests-161%20passed-brightgreen.svg)]()
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%20Cloud%20Ready-FF4B4B.svg)](https://streamlit.io/)

---

## 📌 Executive Summary

**LLM Sentinel** is an automated security evaluation and red-teaming framework designed to audit Large Language Model (LLM) applications and Retrieval-Augmented Generation (RAG) pipelines. It subjects target systems to structured adversarial attacks, detects security breaches using a hybrid evaluation engine (deterministic canary/regex rules + LLM-as-a-Judge), calculates risk scores weighted by severity, and surfaces actionable findings through a comprehensive Streamlit dashboard.

---

## 🎯 Threat Coverage (OWASP Top 10 for LLMs)

LLM Sentinel ships with a curated library of **48 attack cases and 10 benign control cases** (58 total test cases) mapped to key OWASP LLM vulnerability categories:

| OWASP ID | Category | Threat Vector & Test Description | Severity |
| :--- | :--- | :--- | :--- |
| **LLM01** | Prompt Injection | **Direct Jailbreaks**: Persona adoption, roleplay, cognitive bypass.<br>**Indirect Prompt Injection**: Injections embedded in untrusted web pages, forum posts, or emails ingested by RAG.<br>**Position Sensitivity**: Injections systematically tested at `start`, `middle`, and `end` of documents. | High / Critical |
| **LLM02** | Sensitive Information Disclosure | Probing for synthetic confidential student PII, medical records, financial aid data, and planted canary tokens. | Critical |
| **LLM04** | Data & Model Poisoning | Ingestion of poisoned documents carrying false facts to manipulate knowledge retrieval and deceive downstream generation. | Medium / High |
| **LLM07** | System Prompt Leakage | Extraction of hidden developer instructions, system constraints, and secret canaries via translation, summarization, completion, code blocks, and poetry. | High |
| **LLM08** | Excessive Agency & Access Control | Ordinary user role requesting restricted administrative/faculty content from secured documents. | Critical |
| **Controls** | Benign University Inquiries | Harmless questions about the college corpus to quantify the **Control False-Positive Rate** and verify model utility. | Informational |

---

## 🏗️ Architecture & Core Components

```
                            ┌────────────────────────────────────────┐
                            │        Curated Attack Library          │
                            │   (48 Attacks + 10 Benign Controls)    │
                            └───────────────────┬────────────────────┘
                                                │
                                                ▼
┌───────────────────────┐            ┌──────────────────────┐            ┌────────────────────────┐
│  Target RAG Pipeline  │ ◄───────── │ Async Batch Runner   │ ─────────► │     SQLite Storage     │
│ (ChromaDB + Sentence  │            │ (Multi-repeat, limit │            │ (Sessions, Repetitions,│
│  Transformers + LLM)  │            │  & latency tracking) │            │  Retrieved Chunks)     │
└──────────┬────────────┘            └──────────────────────┘            └───────────┬────────────┘
           │                                                                         │
           ▼                                                                         │
┌───────────────────────────────────────────────────┐                                │
│            Hybrid Detection Engine                │                                │
│ ├─ Rule-Based Detector (Canaries & Regex Patterns)│ ───────────────────────────────┤
│ └─ LLM Judge Detector (Semantic Rubric Evaluation)│                                │
└─────────────────────────┬─────────────────────────┘                                │
                          │                                                          │
                          ▼                                                          │
┌───────────────────────────────────────────────────┐                                │
│               Scorer & Risk Engine                │ ◄──────────────────────────────┘
│ ├─ Attack Success Rate (ASR) by Category & Group  │
│ ├─ Weighted Risk Score (0-100, severity weighted) │
│ ├─ Control False-Positive Rate & Run Spread (σ)   │
└─────────────────────────┬─────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                    Interactive Streamlit Security Dashboard                        │
│   (Overview KPIs • Failed Attacks Inspector • Model Comparison • Live Sandbox)     │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Attack Library & Pydantic Schema (`llm_sentinel/attacks/`)
- Strict Pydantic models validating direct, indirect, and poisoning attacks.
- Payload resolution (`{{payload:NAME}}`) and dynamic document synthesis (`build_document()`).
- Automated validation rejecting missing required fields, unknown hosts, or duplicate case IDs.

### 2. Realistic Campus RAG Target (`llm_sentinel/target/rag_app.py`)
- In-memory **ChromaDB** vector database utilizing `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- Synthetic corpus (`data/corpus/`) with 12+ college documents, synthetic confidential records, and restricted documents.
- Naive top-$k$ retrieval simulation demonstrating real-world document ingestion and injection risks.
- Support for `RagTarget` or raw direct wrapper, interfacing with `MockTarget`, local Ollama, or OpenAI endpoints.

### 3. High-Throughput Async Runner (`llm_sentinel/runner/`)
- Runs attacks and controls over $r$ repetitions (default 3) to measure stochastic variance and run-to-run spread.
- Dynamic injection isolation: poisoned documents are indexed, queried, and cleanly unindexed per probe.
- Full SQLite persistence (`reports/sentinel.db`): logs queries, retrieved chunks, model outputs, rule verdicts, judge verdicts, and latencies.

### 4. Hybrid Detection Engine (`llm_sentinel/detectors/`)
- **RuleBasedDetector**: Fast, deterministic regex pattern matching and exact canary token detection.
- **JudgeDetector**: Temperature-0 LLM-as-a-judge with structured JSON output, Rubric guidance, and automated fallback on invalid parsing.

### 5. Quantitative Scorer & Metrics (`llm_sentinel/scorer/`, `llm_sentinel/eval/`)
- **Weighted Risk Score ($0 - 100$)**:
  $$\text{Risk Score} = 100 \times \frac{\sum w_i \cdot s_i}{\sum w_i} \quad \text{where } w \in \{1: \text{low}, 2: \text{medium}, 3: \text{high}, 4: \text{critical}\}$$
- **Attack Success Rate (ASR)** computed per Category, per Attack Group, and Overall.
- **Control False-Positive Rate**: Ensures defensive alignment doesn't break normal application utility.
- **Evaluator Suite**: Confusion matrix, Accuracy, Precision, Recall, F1 score, and Cohen's Kappa ($\kappa$) inter-rater agreement between Rule and Judge detectors.

### 6. Interactive Security Dashboard (`llm_sentinel/dashboard/`)
- Dark-slate security theme built with Streamlit and Plotly.
- Cloud-ready entrypoint (`streamlit_app.py`) for Streamlit Community Cloud.
- 6 comprehensive tabs: Risk Overview, Attack Inspector, Model Comparison, Detector Accuracy, Live Scan Sandbox, and Attack Library Export.

---

## 📁 Repository Structure

```
llm-sentinel/
├── .streamlit/                   # Streamlit dark slate dashboard styling
│   └── config.toml
├── data/
│   ├── corpus/                   # Fictional college RAG corpus & confidential files
│   ├── payloads/                 # Reusable injection payloads
│   ├── seed_attacks.yaml         # Base OWASP attack cases & hosts
│   └── attack_library_additions.yaml # Extended attacks (SL, SD, CP, AC, OB, PP, CT)
│
├── llm_sentinel/                 # Core Python Framework
│   ├── adapters/                 # Model adapters (Mock, Ollama, OpenAI)
│   ├── attacks/                  # Pydantic schemas, validation & library loader
│   ├── dashboard/                # Streamlit Web Application
│   │   ├── app.py                # Main multi-tab UI
│   │   ├── components.py         # Plotly charts, HTML badges & banners
│   │   └── data.py               # Cached data queries & demo seeding
│   ├── detectors/                # Rule-based & LLM-as-a-judge detectors
│   ├── eval/                     # Model evaluation, sampling & metric calculation
│   ├── runner/                   # Engine, SQLite store & session orchestration
│   ├── scorer/                   # Risk score math, ASR calculations & JSON reporter
│   └── target/                   # ChromaDB RAG target implementation
│
├── reports/                      # Scan databases, samples & audit summaries
├── tests/                        # 161 automated pytest test suites
├── requirements.txt              # Production dependencies
├── streamlit_app.py              # Root deployment entrypoint for Streamlit Cloud
├── SPEC.md                       # Architectural specification
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- (Optional) [Ollama](https://ollama.ai/) for running local target models (e.g., `ollama run llama3:8b`)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/llm-sentinel.git
cd llm-sentinel

# Create and activate virtual environment
# Windows:
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation
Run the complete test suite to confirm all components are operational:
```bash
pytest -v
# Output: 161 passed
```

---

## 💻 CLI Usage

### A. Running Security Scans
Execute adversarial probes against a mock target (free, instantaneous) or live model:

```bash
# Free mock run against the College RAG target (3 repetitions, saves to SQLite)
python -m llm_sentinel.runner --target rag --provider mock --model mock-llm --repeats 3 --db reports/sentinel.db

# Live run against a local Ollama model
python -m llm_sentinel.runner --target rag --provider ollama --model llama3:8b --repeats 3 --db reports/sentinel.db

# Quick smoke test with limit
python -m llm_sentinel.runner --target rag --provider mock --limit 5
```

### B. Scoring & Risk Reporting
Calculate ASR, risk scores, and false-positive rates from scan results:

```bash
# Score using hybrid consensus (both rule & judge must agree on SUCCESS)
python -m llm_sentinel.scorer --db reports/sentinel.db --verdict both --report-json reports/summary.json

# Score based strictly on deterministic canary/regex rules
python -m llm_sentinel.scorer --db reports/sentinel.db --verdict rule
```

### C. Detector Evaluation & Human-in-the-Loop Benchmarking
Audit and calibrate detector accuracy:

```bash
# 1. Export stratified sample for manual labeling
python -m llm_sentinel.eval export-sample --n 30 --seed 42 --out reports/label_sample.csv

# 2. Run LLM Judge over unjudged rows
python -m llm_sentinel.eval judge --model gpt-4o-mini --all

# 3. Compute accuracy, confusion matrix, and Cohen's Kappa against ground truth
python -m llm_sentinel.eval metrics --labels reports/label_sample_final.csv
```

---

## 🖥️ Streamlit Dashboard

Launch the interactive security dashboard locally:

```bash
streamlit run streamlit_app.py
```

### Dashboard Features:
1. **Overview & Risk Report**: Model Risk Score gauge, overall ASR, Control False-Positive Rate, Run-to-Run Spread, Plotly horizontal bars for OWASP Category ASR and Group ASR, Attack Type breakdown, and Indirect Injection Position sensitivity analysis (Start vs. Middle vs. End).
2. **Failed Attacks Inspector**: Drill-down card viewer for attacks showing prompt query, retrieved RAG context chunks, model response, rule-based verdict with matched canaries/patterns, and LLM judge verdict & reasoning. Includes a dedicated Control Validation table.
3. **Model & Run Comparison**: Side-by-side KPI comparison across models and scans, grouped ASR comparison charts, and regressed vs. fixed case diffing.
4. **Detector Accuracy**: Rule-based detector vs. LLM Judge confusion matrix, Agreement Rate, Cohen’s Kappa, Accuracy, Precision, Recall, and F1 score.
5. **Live Scan & RAG Sandbox**: Interactive single-case probe and batch scan runner supporting `MockTarget` (free local simulation) or live Ollama models with either campus RAG or raw targets, live progress updates, and ethics disclaimer banner.
6. **Attack Library & Export**: Interactive table of all 48 attack templates and 10 control cases, with one-click JSON and CSV export for downstream reporting.
7. **Offline Demo Seeding**: Includes a **⚡ Seed Demo Runs** button in the sidebar to populate baseline vs. defended scan results instantly without calling external APIs.

---

## ⚖️ Ethics & Controlled Testing Notice

LLM Sentinel is strictly intended for defensive research, academic evaluation, and pre-deployment auditing of AI applications. All adversarial probes and payloads are restricted to local test environments, synthetic corpus fixtures, and reserved `.example` domains. It contains no functional malware or real exploits.

---

## 📄 License

This project is licensed under the Apache 2.0 License. See `LICENSE` for details.
