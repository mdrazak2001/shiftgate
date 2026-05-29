"""
Task registry: load, persist, and manage TaskCluster definitions.

The registry reads from (in priority order):
  1. ``~/.shiftgate/tasks.json``  — user-edited / previously saved
  2. ``shiftgate.data/default_tasks.json``  — bundled defaults (via importlib.resources)

On first run (``shiftgate init``) the ``compute_embeddings`` method is called
to populate ``embedding_centroid`` for every cluster and cache them to
``~/.shiftgate/embeddings_cache.npy`` so subsequent startups are instant.
"""

from __future__ import annotations

import importlib.resources
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from shiftgate.registry.schemas import TaskCluster

if TYPE_CHECKING:
    from shiftgate.router.embedder import Embedder

logger = logging.getLogger(__name__)

# Canonical locations
_SHIFTGATE_DIR = Path.home() / ".shiftgate"
_USER_TASKS_PATH = _SHIFTGATE_DIR / "tasks.json"
_CACHE_PATH = _SHIFTGATE_DIR / "embeddings_cache.npy"

# Legacy dev-checkout path kept for backwards compatibility with source trees
# that still ship repo-root ``data/default_tasks.json``.
_LEGACY_DEFAULT_TASKS_PATH = Path(__file__).parent.parent.parent / "data" / "default_tasks.json"
_BUNDLED_DEFAULT_TASKS_LABEL = "shiftgate.data/default_tasks.json"


def _read_bundled_default_tasks() -> str:
    """Return the bundled default task registry JSON from the installed package."""
    try:
        resource = importlib.resources.files("shiftgate.data") / "default_tasks.json"
        return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        pass

    if _LEGACY_DEFAULT_TASKS_PATH.exists():
        return _LEGACY_DEFAULT_TASKS_PATH.read_text(encoding="utf-8")

    raise FileNotFoundError(_BUNDLED_DEFAULT_TASKS_LABEL)


class TaskRegistry:
    """In-memory store for TaskCluster objects, backed by a JSON file.

    Usage::

        registry = TaskRegistry.load()
        registry.compute_embeddings(embedder)
        registry.save()
    """

    def __init__(self, tasks: list[TaskCluster], source_path: Path) -> None:
        self._tasks: dict[str, TaskCluster] = {t.id: t for t in tasks}
        self._source_path = source_path

    # ------------------------------------------------------------------
    # Factory / persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls) -> "TaskRegistry":
        """Load the task registry from disk.

        Prefers the user's ``~/.shiftgate/tasks.json`` and falls back to the
        bundled ``shiftgate.data/default_tasks.json`` if the user file does not exist.
        """
        if _USER_TASKS_PATH.exists():
            source = _USER_TASKS_PATH
            raw = json.loads(source.read_text(encoding="utf-8"))
        else:
            try:
                raw_text = _read_bundled_default_tasks()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"No task registry found. Expected one of:\n"
                    f"  {_USER_TASKS_PATH}\n"
                    f"  {_BUNDLED_DEFAULT_TASKS_LABEL}\n"
                    "Run `shiftgate init` to set up the default registry."
                ) from None
            source = Path(_BUNDLED_DEFAULT_TASKS_LABEL)
            raw = json.loads(raw_text)

        logger.debug("Loading task registry from %s", source)
        tasks = [TaskCluster.model_validate(t) for t in raw]
        instance = cls(tasks, source_path=source)

        # Eagerly restore cached centroids so ``compute_embeddings`` can be
        # skipped on normal runs (not first init).
        instance._restore_cache()
        return instance

    def save(self) -> None:
        """Persist the current registry to ``~/.shiftgate/tasks.json``."""
        _SHIFTGATE_DIR.mkdir(parents=True, exist_ok=True)
        data = [t.model_dump() for t in self._tasks.values()]
        _USER_TASKS_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("Task registry saved to %s", _USER_TASKS_PATH)

    # ------------------------------------------------------------------
    # Embedding management
    # ------------------------------------------------------------------

    def compute_embeddings(self, embedder: "Embedder") -> None:
        """Compute and store the centroid embedding for every task cluster.

        For each cluster the validation examples are embedded individually and
        then averaged (L2-normalised mean) to form a single centroid vector.
        The results are written back into each ``TaskCluster.embedding_centroid``
        field **and** saved to ``~/.shiftgate/embeddings_cache.npy`` as a
        (n_tasks × dim) float32 array for fast loading on future runs.
        """
        task_list = list(self._tasks.values())
        logger.info("Computing embeddings for %d task clusters…", len(task_list))

        for task in task_list:
            all_examples = task.validation_examples
            embeddings = embedder.embed_batch(all_examples)  # shape: (n, dim)
            centroid = embeddings.mean(axis=0)
            # L2-normalise so cosine similarity reduces to dot product later.
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            task.embedding_centroid = centroid.tolist()

        # Persist centroids as a numpy array indexed by task order.
        self._save_cache(task_list)
        logger.info("Embeddings computed and cached.")

    def _save_cache(self, task_list: list[TaskCluster]) -> None:
        """Write centroids to the numpy cache file."""
        _SHIFTGATE_DIR.mkdir(parents=True, exist_ok=True)
        centroids = [t.embedding_centroid for t in task_list if t.embedding_centroid]
        if centroids:
            arr = np.array(centroids, dtype=np.float32)
            np.save(_CACHE_PATH, arr)
            logger.debug("Centroid cache written to %s (%s)", _CACHE_PATH, arr.shape)

    def _restore_cache(self) -> None:
        """Re-populate ``embedding_centroid`` from the numpy cache if available.

        This avoids a full re-embedding on every startup. The cache is keyed
        positionally — tasks must stay in the same order between runs, which is
        true as long as the registry JSON is not manually reordered.
        """
        if not _CACHE_PATH.exists():
            return
        try:
            arr = np.load(_CACHE_PATH)
            task_list = list(self._tasks.values())
            for i, task in enumerate(task_list):
                if i < len(arr):
                    task.embedding_centroid = arr[i].tolist()
            logger.debug("Restored centroids from cache (%d tasks)", len(task_list))
        except Exception as exc:
            logger.warning("Could not restore embedding cache (%s). Re-run `shiftgate init`.", exc)

    def embeddings_ready(self) -> bool:
        """Return True if all task clusters have a computed centroid."""
        return all(t.embedding_centroid is not None for t in self._tasks.values())

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get_all_tasks(self) -> list[TaskCluster]:
        """Return all registered task clusters."""
        return list(self._tasks.values())

    def get_task(self, task_id: str) -> TaskCluster | None:
        """Return a single task cluster by ID, or None if not found."""
        return self._tasks.get(task_id)

    def add_task(self, task: TaskCluster) -> None:
        """Add or replace a task cluster in the registry.

        If a task with the same ID already exists it is silently overwritten.
        Call ``save()`` afterwards to persist the change.
        """
        self._tasks[task.id] = task
        logger.debug("Task '%s' added to registry.", task.id)

    def remove_task(self, task_id: str) -> bool:
        """Remove a task by ID. Returns True if it existed."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    def __len__(self) -> int:
        return len(self._tasks)
