# 🛡️ LLM Sentinel

Automated security evaluation and vulnerability auditing framework for Large Language Model (LLM) applications.

---

## 📌 Overview

**LLM Sentinel** tests LLM apps, APIs, and pipelines against adversarial prompts, evaluates outputs for security and safety breaches, calculates risk scores, and surfaces actionable findings via an interactive Streamlit dashboard.

### Targeted Security Weaknesses
- **OWASP LLM01**: Prompt Injection (Direct & Indirect)
- **OWASP LLM02**: Sensitive Information Disclosure
- **OWASP LLM07**: System Prompt Leakage
- **Jailbreaks**: Persona adoption, roleplay, and alignment bypass techniques

---

## 🏗️ Architecture & Modules

- **Target Adapter (`llm_sentinel.adapters`)**: Unified client interface to query target LLMs, REST APIs, and pipelines.
- **Attack Library (`attacks/`)**: Declarative YAML test suites categorizing attack payloads and detection rules.
- **Runner (`llm_sentinel.runner`)**: High-throughput async batch probe execution engine.
- **Detector (`llm_sentinel.detectors`)**: Multi-strategy evaluation (regex, canary matching, and LLM-as-a-judge).
- **Scorer (`llm_sentinel.scorer`)**: Severity-weighted risk scoring (0–100) and multi-format report generation.
- **Streamlit Dashboard (`llm_sentinel.dashboard`)**: Interactive web GUI for configuring scans, real-time monitoring, and reviewing audit reports.

---

## 📁 Repository Structure

```
llm-sentinel/
├── .env.example              # Sample environment configuration
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── SPEC.md                   # Technical specification
│
├── attacks/                  # Attack YAML Suites
│   ├── owasp_llm01_injection/
│   ├── owasp_llm02_disclosure/
│   ├── owasp_llm07_leakage/
│   └── jailbreaks/
│
├── llm_sentinel/             # Core Python Package
│   ├── adapters/             # Model / API connection adapters
│   ├── attacks/              # Attack loader & schema validation
│   ├── detectors/            # Attack success detection logic
│   ├── runner/               # Execution orchestration & concurrency
│   ├── scorer/               # Vulnerability scoring & report generation
│   └── dashboard/            # Streamlit dashboard interface
│
├── tests/                    # Automated test suites
└── reports/                  # Generated scan artifacts (JSON, Markdown, CSV)
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ installed

### 2. Virtual Environment Setup
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
Copy `.env.example` to `.env` and set your API keys:
```bash
cp .env.example .env
```

---

## 🧪 Usage (Placeholders)

### CLI Mode (Coming Soon)
```bash
python -m llm_sentinel.runner --config config.yaml
```

### Streamlit Dashboard (Coming Soon)
```bash
streamlit run llm_sentinel/dashboard/app.py
```

---

## 📄 License
This project is licensed under the Apache 2.0 License.
