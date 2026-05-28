# shiftgate ⚡

> **shiftgate is an intelligent routing layer that automatically selects the right LoRA adapter for each task in your local agent loop.**

Instead of hardcoding which adapter to use, shiftgate embeds your query and matches it against a catalog of task clusters using cosine similarity — then routes inference to the best-fit LoRA adapter on your running Ollama or vLLM instance. Think of it as "npm for LoRA adapters + an automatic brain that picks the right one per task."

Inspired by the [LORAUTER paper](https://arxiv.org/abs/2406.08213) (EPFL, 2026).

---

## Quickstart

```bash
# 1. Install (requires Python 3.10+, uv recommended)
uv add shiftgate

# 2. Initialise: sets up ~/.shiftgate/, downloads the embedding model,
#    and computes task centroids
shiftgate init

# 3. Register a LoRA adapter from HuggingFace
shiftgate adapter add monology/pmc-llama-13b-lora --base meta-llama/Meta-Llama-3-8B --tags medical qa

# 4. Route a query (shows decision, no inference needed)
shiftgate route "explain the mechanism of action of ibuprofen"

# 5. Route + run (requires Ollama or vLLM running locally)
shiftgate run "explain the mechanism of action of ibuprofen"
```

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

## Commands

| Command | Description |
|---|---|
| `shiftgate init` | First-time setup: copy defaults to `~/.shiftgate/`, compute embeddings |
| `shiftgate route "<query>"` | Route a query and show the decision — no inference |
| `shiftgate run "<query>"` | Route + run via Ollama or vLLM |
| `shiftgate adapter add <hf_repo>` | Register a new LoRA adapter |
| `shiftgate adapter list` | Table of all registered adapters |
| `shiftgate adapter remove <id>` | Remove an adapter |
| `shiftgate task list` | Table of all task clusters |
| `shiftgate task add` | Interactively add a new task cluster |
| `shiftgate feedback accept` | Mark last routing as good |
| `shiftgate feedback reject` | Mark last routing as bad |
| `shiftgate feedback stats` | Adapter acceptance rate table |
| `shiftgate status` | Backend connectivity + registry summary |
| `shiftgate demo` | Animated demo with fake routing traces |

---

## Using with Ollama

Ollama supports LoRA adapters via custom Modelfiles. Create one per adapter:

```dockerfile
# my-lora.Modelfile
FROM llama3
ADAPTER /path/to/my-adapter.safetensors
```

```bash
ollama create my-lora-model -f my-lora.Modelfile
ollama serve
```

Register the adapter in shiftgate using the same ID:

```bash
shiftgate adapter add my-org/my-lora --base meta-llama/Meta-Llama-3-8B
# The adapter id defaults to the repo slug: "my-lora"
# shiftgate will pass model="my-lora" to Ollama → activates the Modelfile
```

## Using with vLLM

vLLM loads LoRA adapters at startup via `--lora-modules`:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B \
    --enable-lora \
    --lora-modules my-lora=/path/to/adapter
```

shiftgate sends `"model": "<adapter_id>"` in each `/v1/chat/completions` request, which vLLM maps to the named LoRA module.

---

## How to contribute adapters

1. Fork this repo.
2. Add an entry to `data/default_adapters.json` (optional — the registry ships empty by design; adapters are user-managed).
3. Or, better: publish your adapter to HuggingFace and open a PR that documents it in the README's "Community Adapters" section.

To add a task cluster that better matches your domain, edit `data/default_tasks.json` and add `validation_examples` that represent real queries your users ask. Run `shiftgate init` to recompute centroids.

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

| Version | Focus |
|---|---|
| **v0.1** | Single base model, multi-adapter routing ← _current_ |
| v0.2 | Feedback loop + adapter scoring (auto-demote bad adapters) |
| v0.3 | Multi-model routing (route to different base models per task) |
| v1.0 | Community registry + web UI |

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

## License

MIT. See [LICENSE](LICENSE).
