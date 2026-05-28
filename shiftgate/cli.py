"""
shiftgate CLI — Typer-based command-line interface.

All user-facing interactions happen through this module.  Commands are
grouped into four sub-apps (adapter, task, feedback) plus top-level
commands (init, route, run, status, demo).

Entry point (registered in pyproject.toml):
    shiftgate = "shiftgate.cli:app"
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt

from shiftgate.registry.schemas import AdapterEntry, TaskCluster

console = Console()
app = typer.Typer(
    name="shiftgate",
    help="Intelligent LoRA adapter routing for local LLM inference.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

# Sub-apps
adapter_app = typer.Typer(help="Manage the adapter registry.", no_args_is_help=True)
task_app = typer.Typer(help="Manage task clusters.", no_args_is_help=True)
feedback_app = typer.Typer(help="Record and review routing feedback.", no_args_is_help=True)

app.add_typer(adapter_app, name="adapter")
app.add_typer(task_app, name="task")
app.add_typer(feedback_app, name="feedback")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_registries():
    """Return (task_registry, adapter_registry) — exits with error on failure."""
    from shiftgate.registry.adapter_registry import AdapterRegistry
    from shiftgate.registry.task_registry import TaskRegistry

    try:
        task_reg = TaskRegistry.load()
        adapter_reg = AdapterRegistry.load()
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
    return task_reg, adapter_reg


def _get_embedder():
    from shiftgate.router.embedder import Embedder
    return Embedder()


# ---------------------------------------------------------------------------
# shiftgate init
# ---------------------------------------------------------------------------

@app.command()
def init() -> None:
    """Set up ~/.shiftgate/, compute task embeddings, and show a welcome message."""
    from shiftgate.registry.adapter_registry import AdapterRegistry
    from shiftgate.registry.task_registry import TaskRegistry
    from shiftgate.utils.display import show_adapter_table, show_task_table, show_welcome_banner

    show_welcome_banner()

    shiftgate_dir = Path.home() / ".shiftgate"
    shiftgate_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Config directory:[/dim] {shiftgate_dir}")
    console.print()

    # Load defaults (copies them into the user's ~/.shiftgate/ on first save)
    console.print("[cyan]Loading task registry…[/cyan]")
    task_reg = TaskRegistry.load()
    adapter_reg = AdapterRegistry.load()

    # Compute embeddings (downloads model on first run)
    if task_reg.embeddings_ready():
        console.print("[dim]Embeddings already computed. Skipping (delete embeddings_cache.npy to force refresh).[/dim]")
    else:
        console.print("[cyan]Computing task embeddings (first run — model download may take a moment)…[/cyan]")
        embedder = _get_embedder()
        task_reg.compute_embeddings(embedder)
        console.print("[green]✓[/green]  Embeddings computed.")

    # Persist to ~/.shiftgate/
    task_reg.save()
    adapter_reg.save()
    console.print(f"[green]✓[/green]  Registry saved to {shiftgate_dir}")
    console.print()

    show_task_table(task_reg.get_all_tasks())
    console.print()
    show_adapter_table(adapter_reg.list_adapters())
    console.print()

    console.print(
        "[bold green]shiftgate is ready![/bold green]\n\n"
        "  Try it:\n"
        '    [cyan]shiftgate route "write a python function"[/cyan]\n\n'
        "  Add a LoRA adapter:\n"
        "    [cyan]shiftgate adapter add monology/pmc-llama-13b-lora[/cyan]\n"
    )


# ---------------------------------------------------------------------------
# shiftgate adapter
# ---------------------------------------------------------------------------

@adapter_app.command("add")
def adapter_add(
    hf_repo_or_path: Annotated[str, typer.Argument(help="HuggingFace repo ID or local path.")],
    tags: Annotated[
        Optional[list[str]],
        typer.Option("--tags", "-t", help="Task tags, e.g. --tags code python"),
    ] = None,
    base: Annotated[
        Optional[str],
        typer.Option("--base", "-b", help="Base model name, e.g. 'meta-llama/Meta-Llama-3-8B'"),
    ] = None,
    name: Annotated[Optional[str], typer.Option(help="Override display name.")] = None,
    description: Annotated[Optional[str], typer.Option(help="Short description.")] = None,
) -> None:
    """Register a new LoRA adapter from a HuggingFace repo or local path."""
    _, adapter_reg = _load_registries()

    kwargs: dict = {}
    if tags:
        kwargs["tags"] = tags
    if base:
        kwargs["base_model"] = base
    if description:
        kwargs["description"] = description

    with console.status("[cyan]Fetching adapter metadata…[/cyan]"):
        adapter = adapter_reg.add_adapter(hf_repo_or_path, **kwargs)

    if name:
        adapter.name = name
        adapter_reg._adapters[adapter.id] = adapter  # refresh

    adapter_reg.save()
    console.print(f"[green]✓[/green]  Adapter '[bold magenta]{adapter.id}[/bold magenta]' registered.")
    console.print(f"   Name: {adapter.name}")
    console.print(f"   Base: {adapter.base_model}")
    if adapter.task_tags:
        console.print(f"   Tags: {', '.join(adapter.task_tags)}")


@adapter_app.command("list")
def adapter_list() -> None:
    """Show all registered adapters in a Rich table."""
    from shiftgate.utils.display import show_adapter_table

    _, adapter_reg = _load_registries()
    show_adapter_table(adapter_reg.list_adapters())


@adapter_app.command("remove")
def adapter_remove(
    adapter_id: Annotated[str, typer.Argument(help="Adapter ID to remove.")],
) -> None:
    """Remove an adapter from the registry."""
    _, adapter_reg = _load_registries()
    if adapter_reg.remove_adapter(adapter_id):
        adapter_reg.save()
        console.print(f"[green]✓[/green]  Adapter '{adapter_id}' removed.")
    else:
        console.print(f"[red]Error:[/red] Adapter '{adapter_id}' not found.")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# shiftgate task
# ---------------------------------------------------------------------------

@task_app.command("list")
def task_list() -> None:
    """Show all task clusters in a Rich table."""
    from shiftgate.utils.display import show_task_table

    task_reg, _ = _load_registries()
    show_task_table(task_reg.get_all_tasks())


@task_app.command("add")
def task_add() -> None:
    """Interactively add a new task cluster to the registry."""
    task_reg, _ = _load_registries()

    console.print("[bold cyan]Add a new task cluster[/bold cyan]")
    console.print("[dim]Press Ctrl-C to cancel.[/dim]\n")

    task_id = Prompt.ask("Task ID (slug, e.g. code_rust)")
    task_name = Prompt.ask("Display name (e.g. Rust Code Generation)")
    task_desc = Prompt.ask("Short description")

    console.print("\nEnter validation examples (one per line). Empty line to finish:")
    examples: list[str] = []
    while True:
        ex = Prompt.ask(f"  Example {len(examples) + 1} (or Enter to finish)", default="")
        if not ex:
            break
        examples.append(ex)

    if len(examples) < 3:
        console.print("[yellow]Warning:[/yellow] Fewer than 3 examples may produce a poor centroid.")

    preferred_raw = Prompt.ask("Preferred adapter IDs (comma-separated, or leave blank)")
    preferred = [p.strip() for p in preferred_raw.split(",") if p.strip()]

    task = TaskCluster(
        id=task_id,
        name=task_name,
        description=task_desc,
        validation_examples=examples,
        preferred_adapters=preferred,
    )

    if Confirm.ask(f"\nSave task '[bold]{task_id}[/bold]'?"):
        # Recompute centroid for the new task only.
        try:
            embedder = _get_embedder()
            import numpy as np
            vecs = embedder.embed_batch(task.validation_examples)
            centroid = vecs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            task.embedding_centroid = centroid.tolist()
            console.print("[green]✓[/green]  Centroid computed for new task.")
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Could not compute centroid: {exc}")

        task_reg.add_task(task)
        task_reg.save()
        console.print(f"[green]✓[/green]  Task '[bold]{task_id}[/bold]' saved.")
    else:
        console.print("[dim]Cancelled.[/dim]")


# ---------------------------------------------------------------------------
# shiftgate route  — routing only, no inference
# ---------------------------------------------------------------------------

@app.command()
def route(
    query: Annotated[str, typer.Argument(help="Query to route.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of candidate tasks.")] = 3,
    record: Annotated[bool, typer.Option(help="Save trace to ~/.shiftgate/traces.jsonl.")] = True,
) -> None:
    """Route a query to the best adapter (no inference — just the routing decision)."""
    from shiftgate.feedback import loop as feedback_loop
    from shiftgate.registry.task_registry import TaskRegistry
    from shiftgate.router import router as routing
    from shiftgate.utils.display import show_routing_decision

    task_reg, adapter_reg = _load_registries()

    if not task_reg.embeddings_ready():
        console.print("[red]Error:[/red] Task embeddings not initialised. Run `shiftgate init` first.")
        raise typer.Exit(1)

    embedder = _get_embedder()

    try:
        trace = routing.route(query, task_reg, adapter_reg, embedder, top_k=top_k)
    except Exception as exc:
        console.print(f"[red]Routing error:[/red] {exc}")
        raise typer.Exit(1)

    adapter = adapter_reg.get_adapter(trace.selected_adapter_id)
    task = task_reg.get_task(trace.matched_task_id)

    show_routing_decision(
        trace,
        adapter=adapter,
        task_name=task.name if task else None,
        backend_name=None,
    )

    if record:
        feedback_loop.record_trace(trace)
        console.print(f"[dim]Trace {trace.id[:8]}… recorded. Run `shiftgate feedback accept/reject` to rate it.[/dim]")


# ---------------------------------------------------------------------------
# shiftgate run  — route + run inference
# ---------------------------------------------------------------------------

@app.command()
def run(
    query: Annotated[str, typer.Argument(help="Query to route and run.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 3,
) -> None:
    """Route a query and run it through the detected Ollama or vLLM backend."""
    from shiftgate.feedback import loop as feedback_loop
    from shiftgate.router import router as routing
    from shiftgate.runtime.backend import BackendRouter, NoBackendError
    from shiftgate.utils.display import show_routing_decision

    task_reg, adapter_reg = _load_registries()

    if not task_reg.embeddings_ready():
        console.print("[red]Error:[/red] Task embeddings not initialised. Run `shiftgate init` first.")
        raise typer.Exit(1)

    embedder = _get_embedder()

    try:
        trace = routing.route(query, task_reg, adapter_reg, embedder, top_k=top_k)
    except Exception as exc:
        console.print(f"[red]Routing error:[/red] {exc}")
        raise typer.Exit(1)

    adapter = adapter_reg.get_adapter(trace.selected_adapter_id)
    task = task_reg.get_task(trace.matched_task_id)
    backend_router = BackendRouter()
    backend_name = backend_router.detect()

    show_routing_decision(
        trace,
        adapter=adapter,
        task_name=task.name if task else None,
        backend_name=backend_name,
    )

    if adapter is None:
        console.print(f"[red]Adapter '{trace.selected_adapter_id}' not found in registry.[/red]")
        raise typer.Exit(1)

    if backend_name is None:
        console.print(
            "[yellow]No inference backend detected.[/yellow]\n"
            "  shiftgate routed your query to "
            f"[bold magenta]{trace.selected_adapter_id}[/bold magenta].\n"
            "  To run inference, start a backend:\n"
            "    [cyan]ollama serve[/cyan]\n"
            "    [cyan]python -m vllm.entrypoints.openai.api_server --model <base_model>[/cyan]"
        )
        feedback_loop.record_trace(trace)
        raise typer.Exit(0)

    # Run inference
    console.print(f"[cyan]Running via [bold]{backend_name}[/bold]…[/cyan]")
    t0 = time.monotonic()
    try:
        response = backend_router.generate(query, adapter)
    except NoBackendError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Inference error:[/red] {exc}")
        raise typer.Exit(1)

    elapsed_ms = (time.monotonic() - t0) * 1000
    trace.latency_ms = elapsed_ms

    console.print()
    console.rule("[dim]Response[/dim]")
    console.print(response)
    console.rule()
    console.print(f"[dim]Latency: {elapsed_ms:.0f} ms[/dim]")

    feedback_loop.record_trace(trace)


# ---------------------------------------------------------------------------
# shiftgate feedback
# ---------------------------------------------------------------------------

@feedback_app.command("accept")
def feedback_accept() -> None:
    """Mark the last routing trace as accepted (good routing decision)."""
    from shiftgate.feedback import loop as feedback_loop

    trace = feedback_loop.mark_last_accepted(True)
    if trace:
        console.print(f"[green]✓[/green]  Trace {trace.id[:8]}… marked [green]accepted[/green].")
    else:
        console.print("[yellow]No traces found.[/yellow]")


@feedback_app.command("reject")
def feedback_reject() -> None:
    """Mark the last routing trace as rejected (bad routing decision)."""
    from shiftgate.feedback import loop as feedback_loop

    trace = feedback_loop.mark_last_accepted(False)
    if trace:
        console.print(f"[green]✓[/green]  Trace {trace.id[:8]}… marked [red]rejected[/red].")
    else:
        console.print("[yellow]No traces found.[/yellow]")


@feedback_app.command("stats")
def feedback_stats() -> None:
    """Show adapter acceptance rates across all rated traces."""
    from shiftgate.feedback import loop as feedback_loop
    from shiftgate.utils.display import show_feedback_stats

    scores = feedback_loop.compute_adapter_scores()
    stats = feedback_loop.get_trace_stats()
    show_feedback_stats(scores, stats)


# ---------------------------------------------------------------------------
# shiftgate status
# ---------------------------------------------------------------------------

@app.command()
def status() -> None:
    """Show backend connectivity, registry sizes, and embedding status."""
    from shiftgate.runtime.backend import BackendRouter
    from shiftgate.utils.display import show_status

    task_reg, adapter_reg = _load_registries()

    with console.status("[cyan]Probing backends…[/cyan]"):
        backend_router = BackendRouter()
        backend_name = backend_router.detect()

    show_status(
        backend_name=backend_name,
        n_adapters=len(adapter_reg),
        n_tasks=len(task_reg),
        embeddings_ready=task_reg.embeddings_ready(),
    )


# ---------------------------------------------------------------------------
# shiftgate demo
# ---------------------------------------------------------------------------

@app.command()
def demo() -> None:
    """Run an animated demo: fake routing traces and an adapter swap."""
    from shiftgate.registry.schemas import RoutingTrace
    from shiftgate.utils.display import (
        animate_swap,
        show_routing_decision,
        show_welcome_banner,
    )

    show_welcome_banner()
    time.sleep(0.5)

    demo_traces = [
        {
            "query": "Write a Python function to parse JSON from a REST API",
            "task": "Python Code Generation",
            "adapter": "python-lora-llama3",
            "score": 0.91,
        },
        {
            "query": "SELECT all users who signed up in the last 30 days",
            "task": "SQL Query Writing",
            "adapter": "sql-lora-mistral",
            "score": 0.87,
        },
        {
            "query": "Summarise this research paper in 3 bullet points",
            "task": "Text Summarization",
            "adapter": "summarize-lora-llama3",
            "score": 0.83,
        },
        {
            "query": "Fix the KeyError on line 42 in this Python script",
            "task": "Debugging & Error Fixing",
            "adapter": "debug-lora-codellama",
            "score": 0.78,
        },
    ]

    import uuid
    from datetime import datetime, timezone

    for i, entry in enumerate(demo_traces):
        trace = RoutingTrace(
            id=uuid.uuid4().hex,
            query=entry["query"],
            matched_task_id=entry["task"].lower().replace(" ", "_"),
            similarity_score=entry["score"],
            selected_adapter_id=entry["adapter"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        show_routing_decision(trace, task_name=entry["task"])
        time.sleep(0.8)

        if i < len(demo_traces) - 1:
            next_adapter = demo_traces[i + 1]["adapter"]
            animate_swap(entry["adapter"], next_adapter, duration=1.0)
            time.sleep(0.4)

    console.print()
    console.print("[bold green]Demo complete![/bold green]  shiftgate routes tasks at inference time — zero training required.")
    console.print(
        "\n  Get started:\n"
        '    [cyan]shiftgate init[/cyan]\n'
        '    [cyan]shiftgate adapter add <hf_repo>[/cyan]\n'
        '    [cyan]shiftgate route "your query here"[/cyan]\n'
    )
