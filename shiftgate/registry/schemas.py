"""
Pydantic v2 schemas for shiftgate's core data model.

Three top-level types:
  - AdapterEntry  : a LoRA adapter (or fine-tuned model) in the registry
  - TaskCluster   : a group of semantically related tasks with example queries
  - RoutingTrace  : one routing decision, optionally annotated with user feedback
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AdapterEntry(BaseModel):
    """A LoRA adapter registered with shiftgate.

    Adapters can live on HuggingFace (``hf_repo``) or locally (``local_path``).
    At least one of the two must be set for inference to work, though the
    registry itself does not enforce this so adapters can be catalogued before
    they are downloaded.
    """

    id: str = Field(
        description="Unique slug, e.g. 'python-lora-llama3'. Used as a stable reference key."
    )
    name: str = Field(description="Human-readable display name.")
    base_model: str = Field(
        description="The base model this adapter was trained on, e.g. 'meta-llama/Meta-Llama-3-8B'."
    )
    task_tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags describing the adapter's specialisation, e.g. ['code', 'python'].",
    )
    description: str = Field(default="", description="Short prose description of the adapter's purpose.")
    hf_repo: str | None = Field(
        default=None,
        description="HuggingFace Hub repository ID, e.g. 'username/my-lora-adapter'.",
    )
    local_path: str | None = Field(
        default=None,
        description="Absolute path to a local .safetensors file or adapter directory.",
    )
    benchmark_score: float | None = Field(
        default=None,
        description="Optional benchmark score (0–1) reported by the adapter author.",
    )
    context_length: int = Field(
        default=4096,
        description="Maximum context window in tokens.",
    )
    memory_mb: int | None = Field(
        default=None,
        description="Approximate VRAM/RAM usage in MB when the adapter is loaded.",
    )


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
