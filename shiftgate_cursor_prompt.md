# shiftgate — Cursor Agent Implementation Prompt

Paste this entire prompt into Cursor's agent/composer in a blank empty folder.

---

## Prompt

You are implementing **shiftgate** — an open-source CLI tool that acts as an intelligent routing layer for local LLM inference. shiftgate routes agent tasks to the right LoRA adapter (or model) automatically, using task-level semantic matching inspired by the LORAUTER paper (EPFL, 2026). Think of it as "npm for LoRA adapters + an automatic brain that picks the right one per task."

Build the complete v0.1 project from scratch in this folder. Use Python 3.10+. Use `uv` for dependency management (pyproject.toml). Prioritize clean, well-commented code — this is an open-source project that will get contributors.

---

## Project structure to create

```
shiftgate/
├── pyproject.toml
├── README.md
├── .gitignore
├── shiftgate/
│   ├── __init__.py
│   ├── cli.py               # Typer-based CLI entrypoint
│   ├── registry/
│   │   ├── __init__.py
│   │   ├── adapter_registry.py   # CRUD for adapter catalog
│   │   ├── task_registry.py      # Task clusters + embeddings
│   │   └── schemas.py            # Pydantic models for Adapter, Task, RoutingTrace
│   ├── router/
│   │   ├── __init__.py
│   │   ├── embedder.py      # Embeds queries using fastembed
│   │   ├── matcher.py       # Cosine similarity, top-K task retrieval
│   │   └── router.py        # Main routing logic: query → task → adapter
│   ├── runtime/
│   │   ├── __init__.py
│   │   └── backend.py       # Thin client for Ollama and vLLM (OpenAI-compatible)
│   ├── feedback/
│   │   ├── __init__.py
│   │   └── loop.py          # Accept/reject trace storage, adapter scoring
│   └── utils/
│       ├── __init__.py
│       └── display.py       # Rich terminal UI: routing table, adapter swap animation
├── data/
│   ├── default_tasks.json       # Bundled task registry with 10 default task clusters
│   └── default_adapters.json    # 5 example adapter entries pointing to real HF repos
└── tests/
    ├── test_registry.py
    ├── test_router.py
    └── test_feedback.py
```

---

## Detailed implementation spec

### 1. `pyproject.toml`
- Package name: `shiftgate-router`
- CLI entrypoint: `shiftgate = "shiftgate.cli:app"`
- Dependencies: `typer`, `rich`, `pydantic`, `fastembed`, `numpy`, `scikit-learn`, `httpx`, `huggingface-hub`
- Dev dependencies: `pytest`, `pytest-asyncio`

### 2. `shiftgate/registry/schemas.py`
Define these Pydantic v2 models:

```python
class AdapterEntry(BaseModel):
    id: str                    # e.g. "python-lora-llama3"
    name: str
    base_model: str            # e.g. "meta-llama/Meta-Llama-3-8B"
    task_tags: list[str]       # e.g. ["code", "python", "refactor"]
    description: str
    hf_repo: str | None        # HuggingFace repo ID
    local_path: str | None     # Local .safetensors path
    benchmark_score: float | None
    context_length: int = 4096
    memory_mb: int | None

class TaskCluster(BaseModel):
    id: str                    # e.g. "code_python"
    name: str                  # e.g. "Python Code Generation"
    description: str
    validation_examples: list[str]   # 3-10 example queries for this task
    embedding_centroid: list[float] | None = None   # computed at init
    preferred_adapters: list[str]    # adapter IDs in priority order
    fallback_adapters: list[str] = []

class RoutingTrace(BaseModel):
    query: str
    matched_task_id: str
    similarity_score: float
    selected_adapter_id: str
    accepted: bool | None = None   # user feedback
    latency_ms: float | None = None
    timestamp: str
```

### 3. `shiftgate/registry/task_registry.py`
- Load tasks from `~/.shiftgate/tasks.json` (falling back to `data/default_tasks.json`)
- Method `compute_embeddings(model)`: for each task, embed all validation_examples, average them, store as `embedding_centroid`
- Method `get_all_tasks() -> list[TaskCluster]`
- Method `add_task(task: TaskCluster)`
- Method `save()`

### 4. `shiftgate/registry/adapter_registry.py`
- Load adapters from `~/.shiftgate/adapters.json` (falling back to `data/default_adapters.json`)
- Method `get_adapter(id) -> AdapterEntry`
- Method `add_adapter(adapter: AdapterEntry)` — also accepts a HF repo ID string as shorthand
- Method `list_adapters() -> list[AdapterEntry]`
- Method `save()`

### 5. `shiftgate/router/embedder.py`
- Use `fastembed` with model `BAAI/bge-small-en-v1.5` (small, fast, runs on CPU)
- `embed(text: str) -> np.ndarray`
- `embed_batch(texts: list[str]) -> np.ndarray`
- Cache the model as a module-level singleton so it loads once

### 6. `shiftgate/router/matcher.py`
- `top_k_tasks(query_embedding, task_registry, k=3) -> list[tuple[TaskCluster, float]]`
  - computes cosine similarity between query embedding and each task's centroid
  - returns top-K (task, score) pairs sorted descending
- `select_adapter(top_tasks, adapter_registry) -> tuple[AdapterEntry, TaskCluster, float]`
  - takes the highest-scoring task
  - returns its preferred_adapters[0], the task, and the similarity score

### 7. `shiftgate/router/router.py`
- `route(query: str, task_registry, adapter_registry, embedder) -> RoutingTrace`
  - embed query
  - call matcher.top_k_tasks
  - call matcher.select_adapter
  - build and return a RoutingTrace
  - print a Rich panel showing: query, matched task, similarity score, selected adapter

### 8. `shiftgate/runtime/backend.py`
- `class OllamaBackend`: thin httpx client against `http://localhost:11434`
  - `generate(prompt, model_name, adapter_name=None) -> str`
  - Ollama supports LoRA via Modelfile; document this clearly in comments
- `class VLLMBackend`: thin httpx client against `http://localhost:8000`
  - `generate(prompt, model_name, lora_name=None) -> str`
  - use `/v1/chat/completions` with `model` field set to lora_name if provided
- `class BackendRouter`:
  - detects which backend is running (ping both)
  - exposes unified `generate(prompt, adapter: AdapterEntry) -> str`

### 9. `shiftgate/feedback/loop.py`
- Append RoutingTrace as JSON lines to `~/.shiftgate/traces.jsonl`
- `record_trace(trace: RoutingTrace)`
- `mark_accepted(trace_id, accepted: bool)` — updates last N traces
- `compute_adapter_scores() -> dict[str, float]` — acceptance rate per adapter_id

### 10. `shiftgate/utils/display.py`
Use `rich` for all output. Implement:
- `show_routing_decision(trace: RoutingTrace)` — a Rich Panel with:
  - "Query" row
  - "Matched Task" row with similarity % as a colored bar (green >0.8, yellow 0.6-0.8, red <0.6)
  - "Selected Adapter" row with adapter name + base model
  - "Backend" row
- `show_adapter_table(adapters)` — Rich Table listing all registered adapters
- `show_task_table(tasks)` — Rich Table listing all task clusters with their preferred adapters
- `animate_swap(from_adapter, to_adapter)` — a simple Rich Live animation (spinner) showing "Swapping [from] → [to]"

### 11. `shiftgate/cli.py`
Typer app with these commands:

```
shiftgate init                        # sets up ~/.shiftgate/, computes task embeddings, shows welcome
shiftgate adapter add <hf_repo_or_path> [--tags code python] [--base llama3-8b]
shiftgate adapter list                # Rich table of all adapters
shiftgate adapter remove <id>
shiftgate task list                   # Rich table of all task clusters
shiftgate task add                    # interactive prompt to add a new task cluster
shiftgate route "<query>"             # route a query and show decision (no inference, just routing)
shiftgate run "<query>"               # route + actually run via Ollama/vLLM backend
shiftgate feedback accept             # mark last trace as accepted
shiftgate feedback reject             # mark last trace as rejected
shiftgate feedback stats              # show adapter acceptance rates
shiftgate status                      # show backend connectivity, loaded adapters, registry sizes
shiftgate demo                        # a demo, terminal animation and routing trace
```

### 12. `data/default_tasks.json`
Include these 10 task clusters with 5+ validation examples each:
- `code_python` — Python code generation and refactoring
- `code_sql` — SQL query writing
- `code_javascript` — JavaScript/TypeScript
- `code_debug` — Debugging and error fixing (any language)
- `text_summarize` — Summarization of documents/articles
- `text_classify` — Classification and labeling tasks
- `text_translate` — Translation between languages
- `math_reasoning` — Math, arithmetic, word problems
- `qa_factual` — Factual Q&A and knowledge retrieval
- `agent_planning` — Task decomposition and multi-step planning

### 13. `data/default_adapters.json`
Leave default_adapters.json as an empty array []. Document in README that users add adapters via shiftgate adapter add <hf_repo>.

### 14. `README.md`
Write a clean README with:
- One-line description: "shiftgate is an intelligent routing layer that automatically selects the right LoRA adapter for each task in your local agent loop."
- Quickstart (5 commands to go from zero to routing)
- Architecture diagram in ASCII
- How to contribute adapters
- Roadmap section:
  - v0.1: Single base model, multi-adapter routing ← current
  - v0.2: Feedback loop + adapter scoring
  - v0.3: Multi-model routing (route to different base models)
  - v1.0: Community registry + web UI

### 15. `tests/`
Write pytest tests for:
- `test_registry.py`: add/get/list adapters and tasks, save/load round-trip
- `test_router.py`: given a mock task registry with known embeddings, assert correct task is matched for sample queries
- `test_feedback.py`: record trace, mark accepted, assert score updates

---

## Important implementation notes

1. **T4 GPU constraint**: the embedding model (`BAAI/bge-small-en-v1.5` via fastembed) must run on CPU. Do not assume CUDA for the router itself. The backend (Ollama/vLLM) handles GPU inference separately.

2. **No training required**: shiftgate is purely inference-time routing. Zero backpropagation, zero fine-tuning in the router itself. The embedder is a frozen pretrained model.

3. **~/.shiftgate/ layout**:
```
~/.shiftgate/
├── adapters.json
├── tasks.json
├── traces.jsonl
└── embeddings_cache.npy    # cached centroids so init doesn't re-embed on every run
```

4. **First `shiftgate init` should feel magic**: print a welcome banner with Rich, compute embeddings, show a table of default tasks and adapters, and end with "Run `shiftgate route \"write a python function\"` to try it."

5. **Error handling**: if no backend is running, `shiftgate run` should gracefully say "No backend detected. Start Ollama with `ollama serve` or vLLM. shiftgate routed your query to [adapter] — start a backend to run inference." This way the routing logic is testable without a GPU.

6. **Code style**: type hints everywhere, docstrings on all public methods, no global mutable state outside of singletons.

---

Build the complete implementation now. Start with `pyproject.toml` and `schemas.py`, then work through each module in dependency order. After each file, confirm it is complete before moving to the next.
