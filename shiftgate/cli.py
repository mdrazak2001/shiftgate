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


def _auto_link_adapter(adapter: AdapterEntry, task_reg) -> list[str]:
    """Add ``adapter.id`` to the ``preferred_adapters`` of matching task clusters.

    A task cluster matches when at least one of the adapter's ``task_tags``
    appears as a token in the task's ID (e.g. tag ``"sql"`` matches cluster
    ``"code_sql"``).  The adapter is appended only if it is not already listed.

    Returns the list of task IDs that were updated.
    """
    if not adapter.task_tags:
        return []

    adapter_tokens = {t.lower() for t in adapter.task_tags}
    linked: list[str] = []

    for task in task_reg.get_all_tasks():
        task_tokens = set(task.id.lower().split("_"))
        if adapter_tokens & task_tokens:
            if adapter.id not in task.preferred_adapters:
                task.preferred_adapters.append(adapter.id)
                linked.append(task.id)

    return linked


def _finish_adapter_add(adapter: AdapterEntry, task_reg, adapter_reg) -> None:
    """Save registries, print confirmation, and auto-link the adapter to tasks."""
    adapter_reg.save()
    console.print(f"[green]✓[/green]  Adapter '[bold magenta]{adapter.id}[/bold magenta]' registered.")
    console.print(f"   Name:       {adapter.name}")
    console.print(f"   Base model: {adapter.base_model}")
    if adapter.task_tags:
        console.print(f"   Tags:       {', '.join(adapter.task_tags)}")
    if adapter.hf_repo:
        console.print(f"   HF repo:    {adapter.hf_repo}")
    if adapter.local_path:
        console.print(f"   Local path: {adapter.local_path}")
    if adapter.runtime_name:
        console.print(f"   Runtime:    {adapter.runtime_name}")

    linked = _auto_link_adapter(adapter, task_reg)
    if linked:
        # Successfully wired into at least one task cluster → mark routable.
        adapter.status = "linked"
        adapter_reg.save()
        task_reg.save()
        console.print(f"   [green]Status:     linked[/green] → {', '.join(linked)}")
    else:
        console.print(
            "   [yellow]Status:     unassigned[/yellow] — no task clusters matched these tags.\n"
            "   [dim]This adapter will NOT be selected by the router until it is linked. "
            "Use `shiftgate task list` to see clusters.[/dim]"
        )


# ---------------------------------------------------------------------------
# shiftgate init
# ---------------------------------------------------------------------------

@app.command()
def init() -> None:
    """Set up ~/.shiftgate/, compute task embeddings, and show a welcome message."""
    from shiftgate.registry.adapter_registry import AdapterRegistry
    from shiftgate.registry.task_registry import TaskRegistry
    from shiftgate.utils.display import show_task_table, show_welcome_banner

    show_welcome_banner()

    shiftgate_dir = Path.home() / ".shiftgate"
    shiftgate_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Config directory:[/dim] {shiftgate_dir}")
    console.print()

    console.print("[cyan]Loading task registry…[/cyan]")
    task_reg = TaskRegistry.load()

    if task_reg.embeddings_ready():
        console.print("[dim]Embeddings already computed. Skipping (delete embeddings_cache.npy to force refresh).[/dim]")
    else:
        console.print("[cyan]Computing task embeddings (first run — model download may take a moment)…[/cyan]")
        embedder = _get_embedder()
        task_reg.compute_embeddings(embedder)
        console.print("[green]✓[/green]  Embeddings computed.")

    task_reg.save()
    console.print(f"[green]✓[/green]  Task registry saved to {shiftgate_dir}")
    console.print()

    show_task_table(task_reg.get_all_tasks())
    console.print()

    console.print(
        "[bold green]shiftgate is ready![/bold green]\n\n"
        "  Add your first adapter:\n"
        '    [cyan]shiftgate adapter add teknium/sql-lora --tags sql --base llama3[/cyan]\n'
        '    [cyan]shiftgate adapter add my-lora --local /path/to/adapter --tags code[/cyan]\n'
        '    [cyan]shiftgate adapter add my-lora --runtime my-lora-vllm --tags code[/cyan]\n\n'
        "  Then route a query:\n"
        '    [cyan]shiftgate route "write a SQL query"[/cyan]\n'
        '    [cyan]shiftgate route "write a SQL query" --explain[/cyan]\n'
    )


# ---------------------------------------------------------------------------
# shiftgate adapter
# ---------------------------------------------------------------------------

@adapter_app.command("add")
def adapter_add(
    identifier: Annotated[
        str,
        typer.Argument(
            help=(
                "One of: (A) a HuggingFace repo ID (contains '/'), "
                "(B) an adapter slug for a local path (use with --local), "
                "or (C) an adapter slug for a runtime-registered adapter (use with --runtime)."
            )
        ),
    ],
    tags: Annotated[
        Optional[list[str]],
        typer.Option("--tags", "-t", help="Task tags, e.g. --tags sql --tags code"),
    ] = None,
    base: Annotated[
        Optional[str],
        typer.Option("--base", "-b", help="Base model identifier, e.g. 'meta-llama/Meta-Llama-3-8B'"),
    ] = None,
    name: Annotated[Optional[str], typer.Option(help="Override display name.")] = None,
    description: Annotated[Optional[str], typer.Option(help="Short description.")] = None,
    local: Annotated[
        Optional[str],
        typer.Option(
            "--local",
            help="(Mode B) Absolute or relative path to the adapter directory or .safetensors file.",
        ),
    ] = None,
    runtime: Annotated[
        Optional[str],
        typer.Option(
            "--runtime",
            help=(
                "(Mode C) The name the adapter is already registered under in the "
                "running backend (vLLM --lora-modules name, or Ollama model name)."
            ),
        ),
    ] = None,
    adapter_id: Annotated[
        Optional[str],
        typer.Option("--id", help="Override the auto-derived registry ID slug."),
    ] = None,
    benchmark_score: Annotated[
        Optional[float],
        typer.Option("--score", help="Optional quality score between 0 and 1."),
    ] = None,
) -> None:
    """Register a LoRA adapter with shiftgate (metadata only — no weights are downloaded).

    \b
    Three registration modes:

      A) HuggingFace repo  (identifier contains '/')
         shiftgate adapter add teknium/sql-lora --tags sql --base llama3

      B) Local adapter path  (use --local)
         shiftgate adapter add sql-lora --local /models/sql-lora --tags sql --base llama3

      C) Runtime-registered adapter  (use --runtime)
         shiftgate adapter add sql-lora --runtime sql-lora-vllm --tags sql --base llama3
    """
    from shiftgate.registry.adapter_registry import (
        AdapterRegistry,
        adapter_from_hf,
        adapter_from_local,
        adapter_from_runtime,
    )

    task_reg, adapter_reg = _load_registries()

    # Validate: --local and --runtime are mutually exclusive.
    if local and runtime:
        console.print("[red]Error:[/red] --local and --runtime are mutually exclusive.")
        raise typer.Exit(1)

    shared_kwargs: dict = {
        "tags": list(tags) if tags else [],
        "base_model": base or "unknown",
        "description": description,
        "benchmark_score": benchmark_score,
    }
    if name:
        shared_kwargs["name"] = name

    # --- Mode B: local path ---
    if local:
        adapter = adapter_from_local(
            local_path=local,
            adapter_id=adapter_id or identifier,
            **shared_kwargs,
        )

    # --- Mode C: runtime name ---
    elif runtime:
        adapter = adapter_from_runtime(
            runtime_name=runtime,
            adapter_id=adapter_id or identifier,
            **shared_kwargs,
        )

    # --- Mode A: HuggingFace repo (identifier contains '/') ---
    elif "/" in identifier:
        with console.status("[cyan]Reading HuggingFace card metadata (no weights downloaded)…[/cyan]"):
            adapter = adapter_from_hf(
                hf_repo=identifier,
                adapter_id=adapter_id,
                **shared_kwargs,
            )

    # --- Ambiguous: no '/', no --local, no --runtime ---
    else:
        console.print(
            f"[red]Error:[/red] '{identifier}' doesn't look like a HuggingFace repo ID (missing '/').\n"
            "  Use [cyan]--local /path/to/adapter[/cyan] to register a local adapter, or\n"
            "  use [cyan]--runtime <backend-name>[/cyan] for a runtime-registered adapter."
        )
        raise typer.Exit(1)

    adapter_reg.add_adapter(adapter)
    _finish_adapter_add(adapter, task_reg, adapter_reg)


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
    explain: Annotated[
        bool,
        typer.Option("--explain", "-e", help="Show full routing decision tree."),
    ] = False,
    record: Annotated[bool, typer.Option(help="Save trace to ~/.shiftgate/traces.jsonl.")] = True,
) -> None:
    """Route a query to the best adapter (no inference — just the routing decision).

    Use --explain to see the full decision tree: top task matches, similarity
    scores, candidate adapters, and the reason the selected adapter was chosen.
    """
    from shiftgate.feedback import loop as feedback_loop
    from shiftgate.router import router as routing
    from shiftgate.utils.display import show_explain_decision, show_routing_decision

    task_reg, adapter_reg = _load_registries()

    if not task_reg.embeddings_ready():
        console.print("[red]Error:[/red] Task embeddings not initialised. Run `shiftgate init` first.")
        raise typer.Exit(1)

    embedder = _get_embedder()

    try:
        trace, match_result = routing.route(query, task_reg, adapter_reg, embedder, top_k=top_k)
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

    if explain:
        show_explain_decision(trace, match_result)

    if record:
        feedback_loop.record_trace(trace)
        console.print(
            f"[dim]Trace {trace.id[:8]}… recorded. "
            "Run `shiftgate feedback accept/reject` to rate it.[/dim]"
        )


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
        trace, match_result = routing.route(query, task_reg, adapter_reg, embedder, top_k=top_k)
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
        # Either the matched task has no linked adapter, or the linked ID is
        # missing from the registry. In both cases: never guess, never run.
        console.print(
            "[red]No adapter available for this query — not running inference.[/red]\n"
            f"  Matched task: [bold]{trace.matched_task_id}[/bold]\n"
            "  Add one with: "
            f"[cyan]shiftgate adapter add <hf_repo> --tags {trace.matched_task_id}[/cyan]"
        )
        feedback_loop.record_trace(trace)
        raise typer.Exit(1)

    if backend_name is None:
        console.print(
            "[yellow]No inference backend detected.[/yellow]\n"
            "  shiftgate routed your query to "
            f"[bold magenta]{trace.selected_adapter_id}[/bold magenta].\n\n"
            "  Shiftgate is a routing layer — start a backend to run inference:\n"
            "    [cyan]ollama serve[/cyan]\n"
            "    [cyan]python -m vllm.entrypoints.openai.api_server "
            "--model <base_model> --enable-lora[/cyan]"
        )
        feedback_loop.record_trace(trace)
        raise typer.Exit(0)

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
    console.print(
        "[bold green]Demo complete![/bold green]  "
        "shiftgate routes tasks at inference time — zero training required.\n\n"
        "  Shiftgate is a routing layer. You manage models and LoRA weights.\n\n"
        "  Get started:\n"
        '    [cyan]shiftgate init[/cyan]\n'
        '    [cyan]shiftgate adapter add teknium/sql-lora --tags sql --base llama3[/cyan]\n'
        '    [cyan]shiftgate route "your query here" --explain[/cyan]\n'
    )
