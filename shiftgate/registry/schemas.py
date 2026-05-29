"""
Pydantic v2 schemas for shiftgate's core data model.

Three top-level types:
  - AdapterEntry  : metadata record describing a user-managed LoRA adapter
  - TaskCluster   : a group of semantically related tasks with example queries
  - RoutingTrace  : one routing decision, optionally annotated with user feedback

Design note — "Bring Your Own Models"
--------------------------------------
Shiftgate is a *routing layer*.  It stores only metadata; it does not download,
cache, or manage adapter weights.  Users are responsible for making their
weights available to the backend (Ollama, vLLM, etc.) before running inference.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class AdapterEntry(BaseModel):
    """Metadata record for a user-managed LoRA adapter.

    Exactly one of ``hf_repo``, ``local_path``, or ``runtime_name`` should be
    set to describe where the adapter lives relative to the inference backend.
    The registry itself does not enforce this constraint so that adapters can
    be catalogued before they are wired into a backend.

    Source fields
    -------------
    hf_repo
        HuggingFace Hub repo ID (e.g. ``"org/my-lora"``).  Stored for
        informational purposes; shiftgate does not download weights.
    local_path
        Absolute path to a local ``.safetensors`` file or adapter directory
        that the backend can reach directly.
    runtime_name
        The name the adapter is already registered under in the running
        backend (e.g. the ``--lora-modules`` name passed to vLLM, or the
        Ollama model name created from a Modelfile).
    """

    id: str = Field(
        description="Unique slug, e.g. 'sql-lora-llama3'. Used as the stable routing key."
    )
    name: str = Field(description="Human-readable display name.")
    base_model: str = Field(
        description="Base model identifier, e.g. 'meta-llama/Meta-Llama-3-8B'."
    )
    task_tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags describing the adapter's specialisation, e.g. ['code', 'sql'].",
    )
    description: str = Field(
        default="",
        description="Short prose description of the adapter's purpose.",
    )

    # --- source / location fields (at least one should be set) ---
    hf_repo: str | None = Field(
        default=None,
        description="HuggingFace Hub repository ID.  Informational only — shiftgate does not download weights.",
    )
    local_path: str | None = Field(
        default=None,
        description="Absolute path to a local .safetensors file or adapter directory accessible by the backend.",
    )
    runtime_name: str | None = Field(
        default=None,
        description=(
            "Name the adapter is already registered under in the running backend "
            "(e.g. the vLLM --lora-modules name, or the Ollama model name). "
            "shiftgate passes this value to the backend instead of the adapter id "
            "when it is set."
        ),
    )

    # --- optional quality / resource metadata ---
    benchmark_score: float | None = Field(
        default=None,
        description="Optional benchmark score (0–1) reported by the adapter author.",
    )

    @model_validator(mode="after")
    def _at_least_one_source(self) -> "AdapterEntry":
        """Warn (not error) when no source field is set.

        A hard validation error would break backward-compatible loading of
        existing adapters.json files that predate this field.  We log a warning
        instead so old files continue to load cleanly.
        """
        import logging

        if not any([self.hf_repo, self.local_path, self.runtime_name]):
            logging.getLogger(__name__).debug(
                "Adapter '%s' has no hf_repo, local_path, or runtime_name set. "
                "Routing will work but the backend won't know where to find the weights.",
                self.id,
            )
        return self

    def effective_backend_name(self) -> str:
        """Return the name to pass to the backend at inference time.

        Priority: ``runtime_name`` > ``id``.
        ``runtime_name`` is used when the adapter is already registered in the
        backend under a different name than shiftgate's slug.
        """
        return self.runtime_name or self.id


class TaskCluster(BaseModel):
    """A cluster of semantically related tasks used for routing.

    During ``shiftgate init``, the ``validation_examples`` are embedded and
    averaged to produce ``embedding_centroid``.  At routing time the query
    embedding is compared against every cluster's centroid.
    """

    id: str = Field(
        description="Unique slug, e.g. 'code_python'. Used as a stable routing key."
    )
    name: str = Field(description="Human-readable cluster name, e.g. 'Python Code Generation'.")
    description: str = Field(description="Short description of what tasks belong here.")
    validation_examples: list[str] = Field(
        description="3–10 representative query strings used to compute the centroid embedding.",
    )
    embedding_centroid: list[float] | None = Field(
        default=None,
        description="Pre-computed mean embedding of the validation_examples. Populated by init.",
    )
    preferred_adapters: list[str] = Field(
        default_factory=list,
        description="Adapter IDs in priority order. The first available adapter is selected.",
    )
    fallback_adapters: list[str] = Field(
        default_factory=list,
        description="Adapter IDs to try when none of the preferred adapters are available.",
    )


class RoutingTrace(BaseModel):
    """A single routing decision recorded for observability and feedback.

    Traces are appended as JSON lines to ``~/.shiftgate/traces.jsonl``.
    The ``accepted`` field starts as ``None`` and is filled in via
    ``shiftgate feedback accept/reject``.
    """

    id: str = Field(
        description="Unique trace ID (UUID4 hex string) for targeted feedback updates."
    )
    query: str = Field(description="The original user query that triggered this routing decision.")
    matched_task_id: str = Field(description="ID of the TaskCluster that won the similarity match.")
    similarity_score: float = Field(
        description="Cosine similarity between the query embedding and the winning centroid (0–1)."
    )
    selected_adapter_id: str = Field(description="ID of the adapter that was selected for inference.")
    accepted: bool | None = Field(
        default=None,
        description="User feedback: True = good routing, False = bad routing, None = not yet rated.",
    )
    latency_ms: float | None = Field(
        default=None,
        description="End-to-end inference latency in milliseconds (None if only routing, no run).",
    )
    timestamp: str = Field(
        description="ISO-8601 UTC timestamp of when this trace was created."
    )
