# shiftgate ⚡

> **shiftgate is an intelligent routing layer that automatically selects the right LoRA adapter for each task in your local agent loop.**

<p align="center">
  <img src="assets/demo.gif" alt="shiftgate routing a query to the right LoRA adapter" width="720">
</p>

> **shiftgate does not manage weights.** It stores adapter *metadata* only — no downloading, caching, or loading LoRA files. You start **Ollama** or **vLLM** with your models and adapters loaded; shiftgate embeds each query, picks the best task cluster, and tells the backend which adapter to use.

> **`shiftgate run` requires a running inference backend.** Routing-only commands (`shiftgate route`, `shiftgate init`) work without one. To generate text, Ollama (`localhost:11434`) or vLLM (`localhost:8000`) must already be running with your adapters loaded.

Instead of hardcoding which adapter to use, shiftgate matches your query against a catalog of task clusters using cosine similarity — then routes to the best-fit LoRA adapter on that backend.

---

## Quickstart

Requires **Python 3.10+** and a running **Ollama** or **vLLM** instance for inference.

### 1. Install

```bash
uv tool install shiftgate
# or: pip install shiftgate
```

### 2. Start your backend

**vLLM** (example — load adapters with `--lora-modules`):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B \
    --enable-lora \
    --lora-modules python-lora=/path/to/python-lora
```

**Ollama** (example — create a model that bundles base + adapter, then serve):

```bash
ollama create python-lora-ollama -f my-python-lora.Modelfile
ollama serve
```

### 3. Initialise shiftgate

Creates `~/.shiftgate/` and computes task embeddings (one-time model download for routing):

```bash
shiftgate init
```

### 4. Register adapters

Pick the option that matches your setup (see [Bring Your Own Models](#bring-your-own-models) for details):

```bash
# Option 1 — adapter already loaded in vLLM
shiftgate adapter add python-lora --runtime python-lora --tags python --base meta-llama/Meta-Llama-3-8B

# Option 2 — adapter already loaded in Ollama
shiftgate adapter add python-lora --runtime python-lora-ollama --tags python --base llama3

# Option 3 — metadata-only (catalogue a HuggingFace repo; no weights downloaded)
shiftgate adapter add teknium/python-lora --tags python --base llama3
```

### 5. Run a query

```bash
# Route only — shows the decision, no inference
shiftgate route "write a python sorting function"

# Route + run through your backend
shiftgate run "write a python sorting function"
```

**Essential commands:** `init` · `adapter add` · `route` · `run` · `doctor`

---

## Example

```bash
shiftgate run "write a python sorting function"
```

```
╭────────────────────────── Routing Decision ──────────────────────────╮
│  Query          "write a python sorting function"                    │
│  Matched Task   Python Code Generation  ████████████████░░  91.2%    │
│  Adapter        python-lora  [meta-llama/Meta-Llama-3-8B]            │
│  Backend        vllm                                                 │
╰──────────────────────────────────────────────────────────────────────╯

Running via vllm…

────────────────────────────────── Response ──────────────────────────────────
def sort_array(arr):
    """Return a sorted copy using Python's Timsort."""
    return sorted(arr)
───────────────────────────────────────────────────────────────────────────────
Inference: 6204 ms · Total: 6246 ms
```

Use `shiftgate route "<query>" --explain` to see the full decision tree — top task matches, similarity scores, and why an adapter was chosen.

---

## Verify your setup

Run a full health check anytime something feels off:

```bash
shiftgate doctor
```

`shiftgate doctor` checks:

| Check | What it tells you |
| --- | --- |
| **Embedder** | Whether the routing embedding model loads and produces vectors |
| **Backend** | Whether Ollama (`localhost:11434`) or vLLM (`localhost:8000`) is reachable |
| **Task embeddings** | Whether all task clusters have computed centroids (`shiftgate init`) |
| **Adapter runtime availability** | For each registered adapter: linked status and whether it is **loaded** in the backend |
| **Unlinked task clusters** | Task clusters with no adapter wired — routing will match the task but cannot run inference |

**Runtime adapter verification** runs automatically when you register a backend-loaded adapter:

```bash
shiftgate adapter add python-lora --runtime python-lora --tags python --base llama3
#   Backend: vllm ✓ verified        ← adapter found in the running backend
#   Backend: vllm ⚠ runtime 'python-lora' not loaded — did you pass --lora-modules?
#   Backend: not running (verification skipped)
```

**Backend detection** is automatic. `shiftgate run`, `shiftgate status`, and `shiftgate doctor` probe Ollama first, then vLLM. No config file required.

---

## Architecture

```
User query
    │
    ▼
┌──────────────────────────────────────────────────┐
│                   shiftgate CLI                  │
│  shiftgate route / shiftgate run                 │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│                    Router                        │
│                                                  │
│  1. Embed query  (fastembed BAAI/bge-small-en)   │
│  2. Cosine similarity vs task centroids          │
│  3. top-K tasks → walk preferred_adapters list   │
│  4. Return RoutingTrace                          │
└──────────┬───────────────────────┬───────────────┘
           │                       │
           ▼                       ▼
┌─────────────────┐   ┌────────────────────────────┐
│  Task Registry  │   │     Adapter Registry       │
│  ~/.shiftgate/  │   │  ~/.shiftgate/adapters.json│
│  tasks.json     │   │                            │
│  (10 defaults)  │   │  Add via:                  │
└─────────────────┘   │  shiftgate adapter add     │
                      └────────────┬───────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────┐
              │        BackendRouter           │
              │                                │
              │  Ollama  (localhost:11434)     │
              │  vLLM    (localhost:8000)      │
              │  Auto-detected at runtime      │
              └────────────────────────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────┐
              │       Feedback Loop            │
              │  ~/.shiftgate/traces.jsonl     │
              │  shiftgate feedback accept     │
              │  shiftgate feedback stats      │
              └────────────────────────────────┘
```

---

## Bring Your Own Models

shiftgate is a routing layer. **You load weights into Ollama or vLLM first**, then register what you loaded so shiftgate can route to it.

You can also catalogue adapters you have not loaded yet (Option 3) — useful for `shiftgate route`, but `shiftgate run` will not produce output until the adapter is available in a running backend.

### Option 1 — Adapter already loaded in vLLM

Start vLLM with your adapters:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B \
    --enable-lora \
    --lora-modules sql-lora=/path/to/sql-lora
```

Register using the `--lora-modules` key as `--runtime`:

```bash
shiftgate adapter add sql-lora --runtime sql-lora --tags sql --base meta-llama/Meta-Llama-3-8B
```

shiftgate sends `"model": "<runtime_name>"` in each `/v1/chat/completions` request.

### Option 2 — Adapter already loaded in Ollama

Create a Modelfile that bundles your base model and adapter:

```dockerfile
# my-sql-lora.Modelfile
FROM llama3
ADAPTER /path/to/sql-lora.safetensors
```

```bash
ollama create sql-lora-ollama -f my-sql-lora.Modelfile
ollama serve
```

Register using the Ollama model name as `--runtime`:

```bash
shiftgate adapter add sql-lora --runtime sql-lora-ollama --tags sql --base llama3
```

shiftgate passes `runtime_name` (or falls back to `id`) as the Ollama model name.

### Option 3 — Metadata-only registration

Catalogue an adapter without downloading weights — metadata only:

```bash
shiftgate adapter add teknium/sql-lora --tags sql --base llama3
```

You can also record a local path for your own reference (shiftgate still does not load the file):

```bash
shiftgate adapter add sql-lora --local /models/sql-lora --tags sql --base llama3
```

Useful for exploring routing decisions before your backend is set up. To run inference, load the adapter in vLLM or Ollama and re-register with `--runtime`.

---

## How to contribute adapters

1. Fork this repo.
2. Publish your adapter to HuggingFace and open a PR that documents it in a **Community Adapters** section (or add it to your local registry with `shiftgate adapter add`).
3. The adapter registry ships empty by design — adapters are user-managed via `~/.shiftgate/adapters.json`.

To add a task cluster that better matches your domain, run `shiftgate task add` interactively or edit `~/.shiftgate/tasks.json` and add `validation_examples` that represent real queries your users ask. Run `shiftgate init` to recompute centroids.

---

## `~/.shiftgate/` layout

```
~/.shiftgate/
├── adapters.json          # your registered adapters
├── tasks.json             # task clusters (copied from defaults on first init)
├── traces.jsonl           # append-only routing trace log
└── embeddings_cache.npy   # cached centroids — delete to force re-embedding
```

---

## Roadmap

| Version  | Focus                                                         |
| -------- | ------------------------------------------------------------- |
| **v0.1** | Single base model, multi-adapter routing ← *current*          |
| v0.2     | Feedback loop + adapter scoring (auto-demote bad adapters)    |
| v0.3     | Multi-model routing (route to different base models per task) |
| v1.0     | Community registry + web UI                                   |

---

## Development

```bash
# Clone and install in editable mode with all dev dependencies
git clone https://github.com/shiftgate-ai/shiftgate
cd shiftgate
uv sync --extra dev   # creates .venv, installs shiftgate + dev deps

# Run tests (no GPU needed — tests use synthetic embeddings)
uv run pytest

# Run the demo inside the venv
uv run shiftgate demo
```

> **Note:** `uv sync` reads `pyproject.toml` and resolves a locked environment.  
> There is no need to run `pip install` manually. Activate the venv with  
> `.venv/Scripts/activate` (Windows) or `source .venv/bin/activate` (macOS/Linux)  
> if you want the `shiftgate` command on your `PATH` without the `uv run` prefix.

## Releases and Publishing

Releases are managed through a CI release workflow (e.g. GitHub Actions).  
**No manual PyPI API token management is required for normal releases.**

The recommended flow:

1. Bump the version in `pyproject.toml` (`version = "x.y.z"`).
2. Open a PR, get it reviewed and merged.
3. Tag the commit: `git tag vx.y.z && git push origin vx.y.z`.
4. The CI workflow builds the wheel with `uv build` and publishes to PyPI
   using [Trusted Publishing (OIDC)](https://docs.pypi.org/trusted-publishers/)  
   — no stored API token needed.

For a one-off manual publish (maintainers only):

```bash
uv build                    # produces dist/shiftgate-x.y.z-py3-none-any.whl
uv publish                  # authenticates via OIDC or a scoped PyPI token
```

### Project layout

```
shiftgate/
├── cli.py               # Typer CLI — all user commands
├── registry/
│   ├── schemas.py       # Pydantic models: AdapterEntry, TaskCluster, RoutingTrace
│   ├── adapter_registry.py
│   └── task_registry.py
├── router/
│   ├── embedder.py      # fastembed wrapper (CPU, singleton)
│   ├── matcher.py       # cosine similarity, top-K, adapter selection
│   └── router.py        # orchestrates embed → match → trace
├── runtime/
│   └── backend.py       # OllamaBackend, VLLMBackend, BackendRouter
├── feedback/
│   └── loop.py          # trace persistence, accept/reject, scoring
└── utils/
    └── display.py       # Rich panels, tables, animations
```

---

## All commands

| Command                                                  | Description                                                           |
| -------------------------------------------------------- | --------------------------------------------------------------------- |
| `shiftgate init`                                         | First-time setup: initialise `~/.shiftgate/`, compute task embeddings |
| `shiftgate route "<query>"`                              | Route a query and show the decision — no inference                    |
| `shiftgate route "<query>" --explain`                    | Full decision tree: task scores, candidates, selection reason         |
| `shiftgate run "<query>"`                                | Route + run via Ollama or vLLM                                        |
| `shiftgate doctor`                                       | Full health check: embedder, backend, adapters, task embeddings       |
| `shiftgate adapter add <hf_repo> [--tags …] [--base …]`  | Register adapter from HuggingFace (metadata only)                     |
| `shiftgate adapter add <id> --local <path> [--tags …]`   | Register a local adapter path                                         |
| `shiftgate adapter add <id> --runtime <name> [--tags …]` | Register a backend-loaded adapter by its runtime name                 |
| `shiftgate adapter list`                                 | Table of all registered adapters                                      |
| `shiftgate adapter remove <id>`                          | Remove an adapter                                                     |
| `shiftgate task list`                                    | Table of all task clusters                                            |
| `shiftgate task add`                                     | Interactively add a new task cluster                                  |
| `shiftgate feedback accept`                              | Mark last routing as good                                             |
| `shiftgate feedback reject`                              | Mark last routing as bad                                              |
| `shiftgate feedback stats`                               | Adapter acceptance rate table                                         |
| `shiftgate status`                                       | Backend connectivity + registry summary                               |
| `shiftgate demo`                                         | Animated demo with fake routing traces                                |

---

## References

- [LORAUTER](https://arxiv.org/abs/2601.21795) — *Effective LoRA Adapter Routing using Task Representations* (Dhasade et al., EPFL, 2026). shiftgate's task-level semantic routing is inspired by this work; it is not a reimplementation of the paper's full algorithm.

---

## License

MIT. See [LICENSE](LICENSE).
