# shiftgate ⚡

> **shiftgate is an intelligent routing layer that automatically selects the right LoRA adapter for each task in your local agent loop.**

<p align="center">
  <img src="assets/demo.gif" alt="shiftgate routing a query to the right LoRA adapter" width="720">
</p>

**Shiftgate is a routing layer. Users manage models and LoRA weights themselves.**  
shiftgate stores only adapter *metadata* — it never downloads, caches, or manages weights.  
Your inference backend (Ollama, vLLM) is responsible for loading the weights; shiftgate tells it *which* adapter to use for each query.

Instead of hardcoding which adapter to use, shiftgate embeds your query and matches it against a catalog of task clusters using cosine similarity — then routes inference to the best-fit LoRA adapter on your running Ollama or vLLM instance.

---

## Quickstart

Requires **Python 3.10+**.

```bash
# Install
uv tool install shiftgate
# or: pip install shiftgate

# First-time setup — creates ~/.shiftgate/ and computes task embeddings
shiftgate init

# Register an adapter (pick the mode that matches your setup)
shiftgate adapter add teknium/sql-lora --tags sql --base llama3          # HuggingFace metadata
shiftgate adapter add sql-lora --local /models/sql-lora --tags sql --base llama3   # local path
shiftgate adapter add sql-lora --runtime sql-lora-vllm --tags sql --base llama3    # backend-loaded

# Route a query (decision only — no inference)
shiftgate route "write a SQL query to find duplicate rows"

# Route + run (requires Ollama or vLLM running locally)
shiftgate run "write a SQL query to find duplicate rows"
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
│  Matched Task   Python Code Generation  ████████████████░░  91.2%  │
│  Adapter        python-lora-llama3  [meta-llama/Meta-Llama-3-8B]   │
│  Backend        ollama                                               │
╰──────────────────────────────────────────────────────────────────────╯

Running via ollama…

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

**Runtime adapter verification** happens automatically when you register a backend-loaded adapter:

```bash
shiftgate adapter add sql-lora --runtime sql-lora-vllm --tags sql --base llama3
#   Backend: vllm ✓ verified        ← adapter found in the running backend
#   Backend: vllm ⚠ runtime 'sql-lora-vllm' not loaded — did you pass --lora-modules?
#   Backend: not running (verification skipped)
```

**Backend detection** is automatic at runtime. `shiftgate run`, `shiftgate status`, and `shiftgate doctor` probe Ollama first, then vLLM. No config file required.

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
│  Task Registry  │   │     Adapter Registry        │
│  ~/.shiftgate/  │   │  ~/.shiftgate/adapters.json │
│  tasks.json     │   │                            │
│  (10 defaults)  │   │  Add via:                  │
└─────────────────┘   │  shiftgate adapter add     │
                      └────────────┬───────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────┐
              │        BackendRouter           │
              │                                │
              │  Ollama  (localhost:11434)      │
              │  vLLM    (localhost:8000)       │
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

Shiftgate is a routing layer. It stores adapter metadata only.  
**You are responsible for loading weights into your inference backend before running `shiftgate run`.**

### Using with Ollama (Mode B or C)

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

Register in shiftgate using the Ollama model name as `--runtime`:

```bash
# Mode C — backend already has the adapter loaded
shiftgate adapter add sql-lora --runtime sql-lora-ollama --tags sql --base llama3
```

shiftgate passes `runtime_name` (or falls back to `id`) as the Ollama model name.

### Using with vLLM (Mode B or C)

Load adapters at server start with `--lora-modules`:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B \
    --enable-lora \
    --lora-modules sql-lora=/path/to/sql-lora
```

Register in shiftgate:

```bash
# Mode C — adapter name matches the --lora-modules key
shiftgate adapter add sql-lora --runtime sql-lora --tags sql --base meta-llama/Meta-Llama-3-8B
```

shiftgate sends `"model": "<runtime_name>"` in each `/v1/chat/completions` request.

### Registering a HuggingFace adapter (Mode A)

```bash
# Metadata only — no weights downloaded
shiftgate adapter add teknium/sql-lora --tags sql --base llama3
```

This is useful for cataloguing adapters before you have pulled their weights.

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
