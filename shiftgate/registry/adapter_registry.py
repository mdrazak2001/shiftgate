"""
Adapter registry: load, persist, and manage AdapterEntry definitions.

The registry reads from (in priority order):
  1. ``~/.shiftgate/adapters.json``  — user-edited / previously saved
  2. ``<package>/../../data/default_adapters.json``  — bundled defaults (empty list)

Adapters can be added by passing a HuggingFace repo ID string or a full
``AdapterEntry`` object.  When a bare HF repo ID is provided, metadata is
fetched from the Hub to fill in the entry automatically.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from shiftgate.registry.schemas import AdapterEntry

logger = logging.getLogger(__name__)

_SHIFTGATE_DIR = Path.home() / ".shiftgate"
_USER_ADAPTERS_PATH = _SHIFTGATE_DIR / "adapters.json"
_DEFAULT_ADAPTERS_PATH = Path(__file__).parent.parent.parent / "data" / "default_adapters.json"


class AdapterRegistry:
    """In-memory store for AdapterEntry objects, backed by a JSON file.

    Usage::

        registry = AdapterRegistry.load()
        registry.add_adapter(AdapterEntry(id="my-lora", ...))
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
        """Load the adapter registry from disk.

        Prefers ``~/.shiftgate/adapters.json``.  Falls back to the bundled
        ``data/default_adapters.json`` (which ships as an empty list).
        """
        if _USER_ADAPTERS_PATH.exists():
            source = _USER_ADAPTERS_PATH
        elif _DEFAULT_ADAPTERS_PATH.exists():
            source = _DEFAULT_ADAPTERS_PATH
        else:
            logger.warning("No adapter registry found; starting empty.")
            return cls([], source_path=_USER_ADAPTERS_PATH)

        logger.debug("Loading adapter registry from %s", source)
        raw = json.loads(source.read_text(encoding="utf-8"))
        adapters = [AdapterEntry.model_validate(a) for a in raw]
        return cls(adapters, source_path=source)

    def save(self) -> None:
        """Persist the current registry to ``~/.shiftgate/adapters.json``."""
        _SHIFTGATE_DIR.mkdir(parents=True, exist_ok=True)
        data = [a.model_dump() for a in self._adapters.values()]
        _USER_ADAPTERS_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("Adapter registry saved to %s", _USER_ADAPTERS_PATH)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get_adapter(self, adapter_id: str) -> AdapterEntry | None:
        """Return an adapter by ID, or None if not found."""
        return self._adapters.get(adapter_id)

    def list_adapters(self) -> list[AdapterEntry]:
        """Return all registered adapters."""
        return list(self._adapters.values())

    def add_adapter(self, adapter: AdapterEntry | str, **kwargs: object) -> AdapterEntry:
        """Add or replace an adapter in the registry.

        Parameters
        ----------
        adapter:
            Either a fully-constructed ``AdapterEntry`` or a HuggingFace
            repo ID string (e.g. ``"username/my-lora-adapter"``).  When a
            string is provided the repo ID is used as ``hf_repo`` and a
            best-effort ID slug is derived from it.  Extra keyword arguments
            (``tags``, ``base_model``, ``description``) override auto-derived
            values.
        """
        if isinstance(adapter, str):
            adapter = _adapter_from_hf_repo(adapter, **kwargs)

        self._adapters[adapter.id] = adapter
        logger.debug("Adapter '%s' added to registry.", adapter.id)
        return adapter

    def remove_adapter(self, adapter_id: str) -> bool:
        """Remove an adapter by ID. Returns True if it existed."""
        if adapter_id in self._adapters:
            del self._adapters[adapter_id]
            return True
        return False

    def __len__(self) -> int:
        return len(self._adapters)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adapter_from_hf_repo(hf_repo: str, **kwargs: object) -> AdapterEntry:
    """Construct a minimal AdapterEntry from a HuggingFace repo ID.

    Tries to pull card metadata from the Hub.  If that fails (offline, private
    repo, etc.) it builds a stub entry from the repo ID alone.

    Extra ``kwargs`` are merged after auto-detection and override any
    auto-derived fields (``tags``, ``base_model``, ``description``).
    """
    # Derive a clean ID slug from the repo path (e.g. "org/my-lora" → "my-lora")
    slug = hf_repo.split("/")[-1].lower().replace("_", "-")

    entry_data: dict = {
        "id": slug,
        "name": slug.replace("-", " ").title(),
        "base_model": kwargs.pop("base_model", "unknown"),
        "task_tags": kwargs.pop("tags", []),
        "description": kwargs.pop("description", f"Imported from {hf_repo}"),
        "hf_repo": hf_repo,
    }
    entry_data.update(kwargs)

    # Attempt to enrich from HuggingFace Hub metadata.
    try:
        from huggingface_hub import hf_hub_download, model_info  # type: ignore

        info = model_info(hf_repo)
        if info.card_data:
            card = info.card_data
            if hasattr(card, "base_model") and card.base_model:
                base = card.base_model
                entry_data["base_model"] = base[0] if isinstance(base, list) else base
            if hasattr(card, "tags") and card.tags and not entry_data["task_tags"]:
                entry_data["task_tags"] = list(card.tags)[:8]
        if info.id:
            entry_data["name"] = info.id.split("/")[-1]
    except Exception as exc:
        logger.debug("Could not fetch HF metadata for '%s': %s", hf_repo, exc)

    return AdapterEntry.model_validate(entry_data)
