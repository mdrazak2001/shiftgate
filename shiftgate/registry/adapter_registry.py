"""
Adapter registry: load, persist, and manage AdapterEntry metadata.

The registry reads exclusively from ``~/.shiftgate/adapters.json``.
There is no bundled default — adapters are always user-owned.

Shiftgate is a *routing layer*; it stores only metadata.  Users are
responsible for making weights available to their inference backend.

Three registration modes are supported (see module-level helpers):
  - ``adapter_from_hf``      : HuggingFace repo metadata only
  - ``adapter_from_local``   : local .safetensors / adapter directory
  - ``adapter_from_runtime`` : an adapter already loaded in the backend
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from shiftgate.registry.schemas import AdapterEntry

logger = logging.getLogger(__name__)

_SHIFTGATE_DIR = Path.home() / ".shiftgate"
_USER_ADAPTERS_PATH = _SHIFTGATE_DIR / "adapters.json"


class AdapterRegistry:
    """In-memory store for AdapterEntry objects, backed by a JSON file.

    Usage::

        registry = AdapterRegistry.load()
        registry.add_adapter(adapter_from_hf("org/my-lora", tags=["sql"]))
        registry.save()
    """

    def __init__(self, adapters: list[AdapterEntry], source_path: Path) -> None:
        self._adapters: dict[str, AdapterEntry] = {a.id: a for a in adapters}
        self._source_path = source_path

    # ------------------------------------------------------------------
    # Factory / persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls) -> "AdapterRegistry":
        """Load the adapter registry from ``~/.shiftgate/adapters.json``.

        Returns an empty registry when the file does not yet exist (first run
        before the user has added any adapters).  Existing ``adapters.json``
        files written by older shiftgate versions are loaded without error
        because Pydantic ignores unknown fields by default; missing new fields
        (e.g. ``runtime_name``) fall back to ``None``.
        """
        if not _USER_ADAPTERS_PATH.exists():
            logger.debug("No adapters.json found; starting with an empty registry.")
            return cls([], source_path=_USER_ADAPTERS_PATH)

        logger.debug("Loading adapter registry from %s", _USER_ADAPTERS_PATH)
        raw = json.loads(_USER_ADAPTERS_PATH.read_text(encoding="utf-8"))
        adapters = [AdapterEntry.model_validate(a) for a in raw]
        return cls(adapters, source_path=_USER_ADAPTERS_PATH)

    def save(self) -> None:
        """Persist the current registry to ``~/.shiftgate/adapters.json``."""
        _SHIFTGATE_DIR.mkdir(parents=True, exist_ok=True)
        data = [a.model_dump() for a in self._adapters.values()]
        _USER_ADAPTERS_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("Adapter registry saved (%d entries).", len(self._adapters))

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get_adapter(self, adapter_id: str) -> AdapterEntry | None:
        """Return an adapter by ID, or None if not found."""
        return self._adapters.get(adapter_id)

    def list_adapters(self) -> list[AdapterEntry]:
        """Return all registered adapters."""
        return list(self._adapters.values())

    def add_adapter(self, adapter: AdapterEntry) -> AdapterEntry:
        """Add or replace an adapter.  Overwrites silently on duplicate ID."""
        self._adapters[adapter.id] = adapter
        logger.debug("Adapter '%s' added to registry.", adapter.id)
        return adapter

    def remove_adapter(self, adapter_id: str) -> bool:
        """Remove an adapter by ID.  Returns True if it existed."""
        if adapter_id in self._adapters:
            del self._adapters[adapter_id]
            return True
        return False

    def __len__(self) -> int:
        return len(self._adapters)


# ---------------------------------------------------------------------------
# Registration-mode factory functions
# ---------------------------------------------------------------------------

def adapter_from_hf(
    hf_repo: str,
    *,
    adapter_id: str | None = None,
    name: str | None = None,
    base_model: str = "unknown",
    tags: list[str] | None = None,
    description: str | None = None,
    benchmark_score: float | None = None,
) -> AdapterEntry:
    """Build an AdapterEntry from a HuggingFace repo ID.

    Shiftgate stores the ``hf_repo`` string as informational metadata only.
    No weights are downloaded.  The entry is enriched from the Hub's model
    card if the network is reachable; the call degrades gracefully otherwise.

    Parameters
    ----------
    hf_repo:
        HuggingFace Hub repo ID, e.g. ``"teknium/sql-lora"``.
    adapter_id:
        Override the auto-derived slug (default: last path component,
        lowercased with underscores replaced by hyphens).
    name:
        Override the display name.
    base_model:
        Base model identifier.  Overridden by Hub card data when available.
    tags:
        Task tags.  Overridden by Hub card tags when available and ``tags``
        is not supplied.
    description:
        Short description.  Defaults to "HuggingFace: <hf_repo>".
    benchmark_score:
        Optional 0–1 quality score.
    """
    slug = adapter_id or _slugify(hf_repo.split("/")[-1])

    entry: dict = {
        "id": slug,
        "name": name or slug.replace("-", " ").title(),
        "base_model": base_model,
        "task_tags": list(tags or []),
        "description": description or f"HuggingFace: {hf_repo}",
        "hf_repo": hf_repo,
        "benchmark_score": benchmark_score,
    }

    # Best-effort Hub enrichment — no weights are fetched, only card metadata.
    try:
        from huggingface_hub import model_info  # type: ignore

        info = model_info(hf_repo)
        if info.card_data:
            card = info.card_data
            if hasattr(card, "base_model") and card.base_model and base_model == "unknown":
                raw_base = card.base_model
                entry["base_model"] = raw_base[0] if isinstance(raw_base, list) else raw_base
            if hasattr(card, "tags") and card.tags and not entry["task_tags"]:
                entry["task_tags"] = list(card.tags)[:8]
        if info.id and not name:
            entry["name"] = info.id.split("/")[-1]
    except Exception as exc:
        logger.debug("HuggingFace metadata fetch skipped for '%s': %s", hf_repo, exc)

    return AdapterEntry.model_validate(entry)


def adapter_from_local(
    local_path: str,
    *,
    adapter_id: str,
    name: str | None = None,
    base_model: str = "unknown",
    tags: list[str] | None = None,
    description: str | None = None,
    benchmark_score: float | None = None,
) -> AdapterEntry:
    """Build an AdapterEntry pointing to a local adapter directory or file.

    The path is stored as-is; shiftgate does not validate that it exists.
    The backend (Ollama/vLLM) is responsible for loading the weights.

    Parameters
    ----------
    local_path:
        Absolute or relative path to the adapter's directory or
        ``.safetensors`` file.
    adapter_id:
        Unique slug for this adapter in the registry.
    """
    return AdapterEntry(
        id=adapter_id,
        name=name or adapter_id.replace("-", " ").title(),
        base_model=base_model,
        task_tags=list(tags or []),
        description=description or f"Local adapter: {local_path}",
        local_path=local_path,
        benchmark_score=benchmark_score,
    )


def adapter_from_runtime(
    runtime_name: str,
    *,
    adapter_id: str | None = None,
    name: str | None = None,
    base_model: str = "unknown",
    tags: list[str] | None = None,
    description: str | None = None,
    benchmark_score: float | None = None,
) -> AdapterEntry:
    """Build an AdapterEntry for an adapter already loaded in the backend.

    Use this when you have started vLLM with ``--lora-modules my-lora=...``
    or created an Ollama model from a Modelfile, and want to route to it by
    its backend-registered name.

    Parameters
    ----------
    runtime_name:
        The name the backend knows the adapter by (passed as the ``model``
        field to vLLM, or as the Ollama model name).
    adapter_id:
        Slug for shiftgate's registry (defaults to a slugified ``runtime_name``).
    """
    slug = adapter_id or _slugify(runtime_name)
    return AdapterEntry(
        id=slug,
        name=name or slug.replace("-", " ").title(),
        base_model=base_model,
        task_tags=list(tags or []),
        description=description or f"Runtime adapter: {runtime_name}",
        runtime_name=runtime_name,
        benchmark_score=benchmark_score,
    )


def adapter_from_base_model(
    base_model: str,
    *,
    adapter_id: str,
    name: str | None = None,
    tags: list[str] | None = None,
    description: str | None = None,
    benchmark_score: float | None = None,
) -> AdapterEntry:
    """Build an AdapterEntry that routes to a base model with no finetune.

    Used for backends whose base models are always available without any
    upload (e.g. Cloudflare Workers AI ``@cf/`` models).  ``runtime_name`` is
    left unset so the backend runs the base model directly.
    """
    slug = _slugify(adapter_id)
    return AdapterEntry(
        id=slug,
        name=name or slug.replace("-", " ").title(),
        base_model=base_model,
        task_tags=list(tags or []),
        description=description or f"Base model: {base_model}",
        benchmark_score=benchmark_score,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _slugify(s: str) -> str:
    """Convert a string to a lowercase hyphen-separated slug."""
    return s.lower().replace("_", "-").replace(" ", "-")
