# LLM Sentinel: Technical Specification (SPEC.md)

## 1. Executive Overview
**LLM Sentinel** is an automated security evaluation and auditing framework for Large Language Model (LLM) applications. It tests target models and pipelines against known adversarial failure modes, evaluates outputs for compliance and safety breaches, calculates risk scores, and surfaces actionable findings through an interactive Streamlit dashboard.

---

## 2. Scope & Target Vulnerabilities

LLM Sentinel focuses on key risks from the **OWASP Top 10 for LLM Applications** and common alignment vulnerabilities:

1. **OWASP LLM01: Prompt Injection**
   - Direct prompt injection (jailbreak pre-prompt overrides, instruction hijacking).
   - Indirect prompt injection (untrusted external data context overrides).
2. **OWASP LLM02: Sensitive Information Disclosure**
   - Extraction of Personally Identifiable Information (PII), proprietary datasets, API secrets, and internal training snippets.
3. **OWASP LLM07: System Prompt Leakage**
   - Techniques coercing the model to repeat, summarize, encode, or translate its system/developer instructions.
4. **Jailbreaks & Alignment Bypass**
   - Persona adoption (DAN-style, roleplay), linguistic obfuscation (ciphers, base64), hypothetical/fictional framing, and multi-turn escalation.

---

## 3. System Architecture & Modules

```
                    +-----------------------+
                    |  Streamlit Dashboard  |
                    +-----------+-----------+
                                |
                                v
+------------------+    +---------------+    +-------------------+
|  Attack Library  |--->|    Runner     |--->|  Target Adapter   |
|   (YAML Suites)  |    +-------+-------+    | (App / API / LLM) |
+------------------+            |            +---------+---------+
                                |                      |
                                v                      |
                        +---------------+              |
                        |   Detector    |<-------------+
                        +-------+-------+   (Raw Responses)
                                |
                                v
                        +---------------+
                        |    Scorer     |
                        +-------+-------+
                                |
                                v
                         [Audit Report]
```

### 3.1. Target Adapter (`llm_sentinel/adapters/`)
- **Purpose**: Abstract communication with any target LLM or application endpoint.
- **Interfaces**:
  - `BaseAdapter`: Abstract base class defining `send(prompt: str, context: dict | None) -> AdapterResponse`.
  - Concrete implementations:
    - `OpenAIAdapter`: Compatible with OpenAI, Azure OpenAI, and OpenAI-compatible endpoints (vLLM, Ollama, Groq, LiteLLM).
    - `HTTPAdapter`: Custom REST API endpoints accepting JSON payloads.
    - `LangChainAdapter` / `CallableAdapter`: Direct Python function or pipeline invocation for local evaluations.
- **Response Model**: Standardized object returning text, raw response, token counts, and latency.

### 3.2. Attack Library (`llm_sentinel/attacks/` & `attacks/`)
- **Purpose**: Declarative test definitions structured in YAML for maintainability and extensibility without changing code.
- **YAML Schema Components**:
  - `id`: Unique identifier (e.g., `INJ-001`).
  - `name`: Human-readable name.
  - `category`: `prompt_injection`, `sensitive_info`, `system_prompt_leakage`, `jailbreak`.
  - `severity`: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`.
  - `description`: Attack mechanism explanation.
  - `prompts`: List of probe prompts or payload templates.
  - `expected_behavior`: Expected secure response behavior.
  - `detector_rules`: Associated detection heuristics and signatures.

### 3.3. Runner Module (`llm_sentinel/runner/`)
- **Purpose**: Orchestrate scan execution across selected attack suites and target adapters.
- **Features**:
  - Asynchronous / multi-threaded probe execution.
  - Configurable rate limiting, retry backoff, and timeouts.
  - Context and state handling for multi-step prompts.
  - Event streaming (progress callbacks for CLI and UI).

### 3.4. Detector Module (`llm_sentinel/detectors/`)
- **Purpose**: Inspect target outputs to determine whether an attack probe succeeded, failed, or produced inconclusive results.
- **Detection Strategies**:
  - **Pattern / Regex Detector**: Signatures for PII (SSN, emails), API keys, known leak prefixes (`"You are an assistant..."`), or refusal markers (`"I cannot fulfill this request"`).
  - **Keyword & Substring Detector**: Known canary tokens or leaked instruction markers.
  - **Semantic / Embedding Detector**: Cosine similarity against reference leaked texts or known jailbreak success patterns.
  - **LLM-as-a-Judge**: Evaluator LLM prompting to grade whether output represents a policy violation or system leakage.

### 3.5. Scorer & Reporting Module (`llm_sentinel/scorer/`)
- **Purpose**: Aggregate evaluation results into structured risk scores and compliance metrics.
- **Features**:
  - **Risk Score Formulation**: Normalized 0–100 risk score based on severity weights (`CRITICAL` > `HIGH` > `MEDIUM` > `LOW`) and exploit success rates.
  - **Vulnerability Breakdown**: Pass/Fail metrics mapped by OWASP category.
  - **Exporter**: Export audit results to JSON, Markdown, and CSV.

### 3.6. Streamlit Dashboard (`llm_sentinel/dashboard/`)
- **Purpose**: User-friendly GUI for launching scans, monitoring progress, and reviewing findings.
- **Views**:
  - **Scan Configuration**: Select target adapter, credentials, endpoint URL, and attack suites.
  - **Live Execution View**: Progress bar, latency metrics, and real-time pass/fail stream.
  - **Audit Overview**: High-level risk score gauge, OWASP vulnerability distribution charts, and severity breakdowns.
  - **Probe Inspector**: Detailed table filtering by vulnerability type, with full side-by-side prompt and model response inspection.
  - **Report Export**: One-click download of audit reports.

---

## 4. Requirements Specification

### 4.1. Functional Requirements (FR)
- **FR-01**: Must support pluggable adapters with a unified async and sync interface.
- **FR-02**: Must load and validate attack suites from YAML files matching a strict schema.
- **FR-03**: Must support variable templating in attack prompts (e.g., target names, canary tokens).
- **FR-04**: Must support multiple detection mechanisms per attack probe (hybrid detection).
- **FR-05**: Must output deterministic results with reproducible seeds and execution logs.
- **FR-06**: Must provide a web-based dashboard via Streamlit to operate the tool without CLI intervention.

### 4.2. Non-Functional Requirements (NFR)
- **NFR-01 (Safety)**: All sample attack payloads in repository suites must be scoped strictly for evaluation and benchmarking; dangerous remote code execution payloads or weaponized malware generation are out of scope.
- **NFR-02 (Extensibility)**: New attack types, detection rules, and adapters must be implementable by adding self-contained modules without altering core runner logic.
- **NFR-03 (Performance)**: Scan runner must support concurrency to execute batches of 100+ probes efficiently within API rate limits.
- **NFR-04 (Traceability)**: Every evaluated response must retain full telemetry: input prompt, raw response, timestamp, latency, adapter parameters, and detector verdict.

---

## 5. Proposed Folder Structure

```
llm-sentinel/
├── .env.example                     # Sample environment variable template
├── .gitignore                       # Git ignore rules (env, cache, venv, results)
├── pyproject.toml                   # Project metadata and dependencies
├── README.md                        # Documentation and quickstart guide
├── SPEC.md                          # Technical specification (this document)
│
├── attacks/                         # Attack YAML Suites
│   ├── owasp_llm01_injection/       # Prompt injection test suites
│   │   ├── direct_injection.yaml
│   │   └── indirect_injection.yaml
│   ├── owasp_llm02_disclosure/      # Sensitive info disclosure suites
│   │   ├── pii_leakage.yaml
│   │   └── credential_extraction.yaml
│   ├── owasp_llm07_leakage/         # System prompt leakage suites
│   │   ├── system_prompt_extraction.yaml
│   │   └── instruction_override.yaml
│   └── jailbreaks/                  # Alignment bypass & persona attacks
│       ├── persona_adoption.yaml
│       └── hypothetical_scenarios.yaml
│
├── llm_sentinel/                    # Core Python Package
│   ├── __init__.py
│   ├── config.py                    # Global configuration and settings loader
│   │
│   ├── adapters/                    # Target Model Adapters
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseAdapter abstract class
│   │   ├── http.py                  # Generic REST API adapter
│   │   └── openai_adapter.py        # OpenAI / compatible endpoint adapter
│   │
│   ├── attacks/                     # Attack Suite Loader & Models
│   │   ├── __init__.py
│   │   ├── models.py                # Pydantic models for attack schemas
│   │   └── loader.py                # YAML parser and validator
│   │
│   ├── detectors/                   # Vulnerability Detectors
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseDetector abstract class
│   │   ├── regex.py                 # Pattern and signature detector
│   │   ├── canary.py                # Canary token detector
│   │   └── judge.py                 # LLM-as-a-judge detector
│   │
│   ├── runner/                      # Test Execution Engine
│   │   ├── __init__.py
│   │   ├── engine.py                # Async batch execution runner
│   │   └── session.py               # Test session tracking & event emission
│   │
│   ├── scorer/                      # Scoring & Risk Assessment
│   │   ├── __init__.py
│   │   ├── metrics.py               # Risk scoring algorithms and weights
│   │   └── report.py                # Report generator (JSON, MD, CSV)
│   │
│   └── dashboard/                   # Streamlit Web Interface
│       ├── __init__.py
│       ├── app.py                   # Main Streamlit entrypoint
│       ├── components/              # UI widgets, metric cards, charts
│       └── views/                   # Tab views (Config, Live, Results)
│
├── tests/                           # Unit and Integration Tests
│   ├── test_adapters.py
│   ├── test_loader.py
│   ├── test_detectors.py
│   └── test_runner.py
│
└── reports/                         # Default output directory for scan artifacts
```

---

## 6. Implementation Milestones

- **Phase 1**: Project scaffolding, Pydantic attack schemas, and YAML loader.
- **Phase 2**: Target adapters (`BaseAdapter`, `OpenAIAdapter`, `HTTPAdapter`) and async runner engine.
- **Phase 3**: Detector suite (regex, canary tokens, refusal detection, and judge model).
- **Phase 4**: Scorer metrics calculation and report generation.
- **Phase 5**: Streamlit interactive dashboard and visualizations.
