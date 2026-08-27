# InterviewLens

**Agentic, resume-aware interview question generator.**

> Takes a Job Description + Candidate Resume → generates targeted, annotated interview questions using an explicit PERCEIVE → REASON → ACT → REFLECT agentic loop.

---

## Quick Start

```bash
# 1. Clone and enter the project
cd interview-agent

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here

# 5. Run with sample data
python main.py

# 6. Run with your own files
python main.py --jd path/to/jd.txt --resume path/to/resume.txt

# 7. Save output to JSON
python main.py --output output/questions.json

# 8. Override iteration limit
python main.py --max-iterations 5
```

---

## What It Does

Given **only** a Job Description and a Candidate Resume, InterviewLens:

- Extracts structured requirements from the JD
- Finds relevant resume evidence for each requirement
- Generates interview questions in three categories:
  - **JD-based** — derived from role requirements
  - **Resume-based** — probing specific candidate experience
  - **Combined** — connecting a JD requirement with resume evidence
- Annotates every question with: category, target skill, relevance, resume connection, and answer hint
- Reflects on each question using a quality rubric before accepting it

**It does NOT:** conduct a live interview, evaluate candidate responses, assign scores, or make hire/reject decisions.

---

## Architecture

```
main.py
  └── AgentLoop.run(jd, resume)
        ├── PERCEIVE  → agent/perceive.py  → PerceptionContext (+ memory.recall)
        ├── REASON    → agent/reason.py    → ReasoningDecision (ReAct pattern)
        ├── ACT       → agent/act.py       → ActResult (tool call + Gemini generation)
        └── REFLECT   → agent/reflect.py   → ReflectionResult (Reflexion pattern + memory.save)
```

**Reasoning pattern:** ReAct (Thought → Action → Observation → Decision)  
**Quality control:** Reflexion (self-critique rubric, retry with targeted feedback)  
**Memory:** Gemini Embeddings + FAISS + JSON metadata (`memory/`)  
**Tools:** `extract_requirements`, `find_resume_evidence`

---

## Project Structure

```
interview-agent/
├── main.py                  # Entry point
├── config.yaml              # All runtime parameters
├── .env.example             # API key template
├── requirements.txt
├── agent/                   # Four phase functions + state
│   ├── loop.py              # AgentLoop orchestrator
│   ├── perceive.py          # PERCEIVE phase
│   ├── reason.py            # REASON phase (ReAct)
│   ├── act.py               # ACT phase
│   ├── reflect.py           # REFLECT phase (Reflexion)
│   └── state.py             # AgentState dataclass
├── tools/                   # Core tools
│   ├── extract_requirements.py
│   ├── find_resume_evidence.py
│   └── schemas.py
├── memory/                  # Semantic memory
│   ├── memory.py            # SemanticMemory: save/recall/clear
│   ├── vector_store.py      # FAISS IndexFlatL2
│   ├── embeddings.py        # Gemini embedding helpers
│   └── metadata.json        # Persisted metadata
├── llm/
│   └── gemini_client.py     # GeminiClient with retry + token tracking
├── harness/                 # Reliability
│   ├── retry.py             # Exponential backoff + jitter
│   ├── fallback.py          # Fallback handlers
│   ├── guardrails.py        # Iteration cap, token budget, stuck-loop
│   └── logger.py            # Structured JSON logger
├── prompts/                 # Externalised prompt templates
│   ├── reasoning.txt        # ReAct reasoning prompt
│   ├── generation.txt       # Question generation prompt
│   └── reflection.txt       # Reflexion rubric prompt
├── tests/                   # pytest test suite
│   ├── test_memory.py
│   ├── test_tools.py
│   ├── test_reflection.py
│   └── test_guardrails.py
└── data/
    ├── sample_jd.txt
    └── sample_resume.txt
```

---

## Configuration

All runtime parameters are in `config.yaml` — nothing is hardcoded in Python:

```yaml
agent:
  max_iterations: 20          # Hard iteration cap
  max_retries_per_question: 3 # Reflexion retry limit
  stuck_loop_window: 3        # Halt if no progress for N iterations
  token_budget: 50000         # Hard token cap

llm:
  model: "gemini-3.5-flash-lite"
  embedding_model: "models/text-embedding-004"
  temperature: 0.7

memory:
  top_k: 3                    # Recall top-K entries per query

retry:
  max_attempts: 3
  base_delay_s: 1.0
  max_delay_s: 30.0
```

---

## Memory Design

SemanticMemory uses FAISS (in-process, exact L2) + Gemini embeddings.

**Why semantic memory in a single-session system?**

Even processing one JD and one resume, memory is valuable *within* the run:
- `reason()` recalls covered requirements → avoids duplicate questions
- `reason()` recalls prior reflection feedback → avoids repeating quality failures
- `reflect()` saves accepted questions → informs de-duplication checks

**Memory flow:**
- `reason()` → `memory.recall(target_requirement)` before deciding
- `reflect()` → `memory.save(question + feedback)` after evaluation
- `main.py` → `memory.clear()` at the start of every run

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use mock Gemini clients — no API key required for testing.

Key test: `test_memory.py::TestMemoryInfluencesReasoning::test_memory_causes_different_requirement_selection`  
Demonstrates that saving a "covered_requirement" entry to memory causes the
simulated `reason()` function to select a different (uncovered) requirement.

---

## Tech Stack

| Technology | Role |
|---|---|
| Python 3.10+ | Agent orchestration |
| Gemini API (`gemini-2.0-flash`) | Reasoning, generation, reflection |
| Gemini Embeddings (`text-embedding-004`) | Semantic memory |
| FAISS (faiss-cpu) | In-process vector similarity search |
| JSON file | Metadata persistence |
| python-dotenv | Secret management |
| PyYAML | Config loading |
| pytest | Test suite |

**Deliberately excluded:** LangChain, CrewAI, AutoGen, LlamaIndex, any external vector DB.

---

## Sample Output

```
======================================================================
  INTERVIEWER LENS — 3 Question(s) Generated
======================================================================

── Question 1 ──────────────────────────────────────────────────────
  Type      : COMBINED
  Category  : Technical
  Skill     : Distributed Systems Design

  Q: You migrated a monolith to 14 microservices at Acme Corp. How would
     you extend that architecture to handle a multi-region active-active
     topology for a new service here?

  Relevance : Tests distributed systems depth — a core JD requirement.
  Resume ↗  : "Deployed all services on GKE across 3 regions"
  Hint      : Draw on your GKE multi-region experience. Discuss leader
              election, replication strategies, and latency trade-offs
              from what you actually implemented.

── Question 2 ──────────────────────────────────────────────────────
  Type      : RESUME_BASED
  Category  : Technical
  Skill     : Performance Engineering

  Q: You reduced p99 latency by 40% at Acme. What was the root cause of
     the latency and what specific change drove the largest improvement?

  Relevance : Probes the depth of a self-reported performance achievement.
  Resume ↗  : "reducing p99 latency by 40%"
  Hint      : Describe your profiling approach, whether the bottleneck was
              I/O, queries, or serialisation, and what change had the
              most impact.
```

---

## Reliability Features

| Feature | Implementation |
|---|---|
| Retry + backoff + jitter | `harness/retry.py::with_retry()` |
| Fallback: bad LLM output | `harness/fallback.py::handle_malformed_generation()` |
| Fallback: tool failure | `harness/fallback.py::handle_*_failure()` |
| Fallback: memory failure | `harness/fallback.py::handle_memory_*_failure()` |
| Hard iteration cap | `harness/guardrails.py::check_iteration_limit()` |
| Token budget | `harness/guardrails.py::check_token_budget()` |
| Stuck-loop detection | `harness/guardrails.py::check_stuck_loop()` |
| Structured JSON logs | `harness/logger.py::log_step()` |
| Config-driven | `config.yaml` — all parameters |
