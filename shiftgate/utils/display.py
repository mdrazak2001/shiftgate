"""
Rich terminal UI helpers for shiftgate.

All console output in shiftgate goes through this module so that visual style
is centralised and easy to update.  Each function accepts domain objects
directly — never raw strings — so the display layer stays decoupled from
formatting decisions.
"""

from __future__ import annotations

import time

from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from shiftgate.registry.schemas import AdapterEntry, RoutingTrace, TaskCluster

console = Console()


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _similarity_colour(score: float) -> str:
    """Return a colour name based on similarity score thresholds."""
    if score >= 0.80:
        return "green"
    if score >= 0.60:
        return "yellow"
    return "red"


def _similarity_bar(score: float, width: int = 20) -> Text:
    """Build a coloured progress bar for the similarity score."""
    filled = round(score * width)
    bar = "█" * filled + "░" * (width - filled)
    colour = _similarity_colour(score)
    text = Text()
    text.append(bar, style=colour)
    text.append(f"  {score * 100:.1f}%", style=f"bold {colour}")
    return text


def _adapter_source_label(adapter: AdapterEntry) -> str:
    """Return a short label describing where the adapter lives."""
    if adapter.runtime_name:
        return f"runtime:{adapter.runtime_name}"
    if adapter.hf_repo:
        return f"hf:{adapter.hf_repo}"
    if adapter.local_path:
        return f"local:{adapter.local_path}"
    return "—"


# ---------------------------------------------------------------------------
# Routing decision panel
# ---------------------------------------------------------------------------

def show_routing_decision(
    trace: RoutingTrace,
    adapter: AdapterEntry | None = None,
    task_name: str | None = None,
    backend_name: str | None = None,
    loaded_runtimes: set[str] | None = None,
    selection_method: str | None = None,
) -> None:
    """Print a Rich Panel describing a routing decision.

    Parameters
    ----------
    trace:
        The ``RoutingTrace`` returned by the router.
    adapter:
        Optional ``AdapterEntry`` for richer adapter display.
    task_name:
        Human-readable task cluster name (falls back to trace.matched_task_id).
    backend_name:
        Active backend name ('ollama', 'vllm', 'cerebras', or None).
    loaded_runtimes:
        Optional set of runtime names loaded on the active backend (used to
        explain a ``no_adapter_on_active_backend`` outcome).
    selection_method:
        The ``MatchResult.selection_method`` (e.g. ``"no_adapter_for_task"`` or
        ``"no_adapter_on_active_backend"``) used to tailor the no-adapter help.
    """
    # When no adapter was selected the decision is unactionable — render red
    # regardless of how confident the task match was.
    no_adapter = adapter is None and trace.selected_adapter_id is None
    colour = "red" if no_adapter else _similarity_colour(trace.similarity_score)

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", min_width=16)
    grid.add_column()

    grid.add_row("Query", Text(f'"{trace.query}"', style="italic cyan"))

    task_display = task_name or trace.matched_task_id
    task_text = Text()
    task_text.append(task_display, style="bold white")
    task_text.append("  ")
    task_text.append_text(_similarity_bar(trace.similarity_score))
    grid.add_row("Matched Task", task_text)

    if no_adapter and selection_method == "no_adapter_on_active_backend":
        # Adapters are linked to this task but none are loaded on the active
        # backend — a different, backend-specific message.
        grid.add_row(
            "Adapter",
            Text(
                f"No adapter loaded on backend '{backend_name or 'unknown'}'",
                style="bold red",
            ),
        )
        runtimes = sorted(loaded_runtimes) if loaded_runtimes else []
        runtimes_label = ", ".join(runtimes) if runtimes else "(none)"
        grid.add_row("Loaded runtimes", Text(runtimes_label, style="dim"))
        suggestion = Text()
        suggestion.append("Try ", style="dim")
        suggestion.append("shiftgate adapter list", style="cyan")
        suggestion.append(" to see what's registered.", style="dim")
        grid.add_row("Suggestion", suggestion)
    elif no_adapter:
        # Never silently substitute an adapter. Tell the user how to fix it.
        adapter_text = Text("No adapter available", style="bold red")
        grid.add_row("Adapter", adapter_text)
        suggestion = Text()
        suggestion.append("Add one with: ", style="dim")
        suggestion.append(
            f"shiftgate adapter add <hf_repo> --tags {trace.matched_task_id}",
            style="cyan",
        )
        grid.add_row("Suggestion", suggestion)
    elif adapter:
        adapter_text = Text()
        adapter_text.append(adapter.name, style="bold magenta")
        adapter_text.append(f"  [{adapter.base_model}]", style="dim")
        source = _adapter_source_label(adapter)
        if source != "—":
            adapter_text.append(f"\n  {source}", style="dim blue")
        grid.add_row("Adapter", adapter_text)
    else:
        grid.add_row("Adapter", Text(str(trace.selected_adapter_id), style="bold magenta"))

    backend_text = Text(backend_name or "—", style="green" if backend_name else "dim")
    grid.add_row("Backend", backend_text)

    if trace.latency_ms is not None:
        grid.add_row("Latency", Text(f"{trace.latency_ms:.0f} ms", style="dim"))

    title = (
        "[bold red] no adapter available [/bold red]"
        if no_adapter
        else f"[bold {colour}] shiftgate routing decision [/bold {colour}]"
    )
    panel = Panel(grid, title=title, border_style=colour, expand=False)
    console.print()
    console.print(panel)
    console.print()


# ---------------------------------------------------------------------------
# --explain view
# ---------------------------------------------------------------------------

def show_explain_decision(
    trace: RoutingTrace,
    match_result,  # MatchResult — avoid circular import
    adapter_registry=None,
    task_registry=None,
) -> None:
    """Print the full routing decision tree for ``shiftgate route --explain``.

    Shows:
      • Top task matches with similarity scores
      • Candidate adapters found for each task
      • Which adapter was selected and why
    """
    from shiftgate.router.matcher import MatchResult  # local import avoids circularity

    console.print()
    console.rule("[bold cyan]Routing explanation[/bold cyan]")
    console.print()

    # --- Query ---
    console.print(f'  [dim]Query:[/dim]  [italic cyan]"{trace.query}"[/italic cyan]')
    console.print()

    # --- Task match table ---
    task_table = Table(
        title="Task similarity ranking",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        expand=False,
    )
    task_table.add_column("#", justify="right", style="dim", width=3)
    task_table.add_column("Task ID", style="yellow")
    task_table.add_column("Task Name")
    task_table.add_column("Similarity", justify="right")
    task_table.add_column("Score bar")
    task_table.add_column("Candidate Adapters", style="magenta")

    for rank, tm in enumerate(match_result.all_task_matches, start=1):
        is_winner = tm.task.id == trace.matched_task_id
        rank_str = f"[bold green]▶ {rank}[/bold green]" if is_winner else str(rank)
        score_pct = f"{tm.score * 100:.1f}%"
        candidates = (
            ", ".join(a.id for a in tm.candidate_adapters) if tm.candidate_adapters else "[dim]none[/dim]"
        )
        task_table.add_row(
            rank_str,
            tm.task.id,
            tm.task.name,
            score_pct,
            _similarity_bar(tm.score, width=12),
            candidates,
        )

    console.print(task_table)
    console.print()

    # --- Selection summary ---
    method_labels = {
        "preferred": "[green]preferred_adapters list[/green]",
        "fallback": "[yellow]fallback_adapters list[/yellow]",
        "no_adapter_for_task": "[red]no adapter linked to the matched task[/red]",
    }
    method_display = method_labels.get(match_result.selection_method, match_result.selection_method)

    selected = match_result.selected_adapter
    if selected is None:
        console.print("  [bold]Selected adapter:[/bold]  [bold red]No adapter available[/bold red]")
        console.print(f"  [bold]Selection method:[/bold]  {method_display}")
        console.print(
            "  [dim]Add one with:[/dim]  "
            f"[cyan]shiftgate adapter add <hf_repo> --tags {match_result.matched_task.id}[/cyan]"
        )
    else:
        console.print(f"  [bold]Selected adapter:[/bold]  [bold magenta]{selected.id}[/bold magenta]")
        console.print(f"  [bold]Base model:[/bold]        {selected.base_model}")
        console.print(f"  [bold]Source:[/bold]            {_adapter_source_label(selected)}")
        console.print(f"  [bold]Selection method:[/bold]  {method_display}")
    console.print()
    console.rule()
    console.print()


# ---------------------------------------------------------------------------
# Adapter table
# ---------------------------------------------------------------------------

def show_adapter_table(adapters: list[AdapterEntry]) -> None:
    """Print a Rich table listing all registered adapters."""
    if not adapters:
        console.print(
            "[dim]No adapters registered.\n"
            "  Add one:  shiftgate adapter add <hf_repo>\n"
            "            shiftgate adapter add <id> --local /path/to/adapter\n"
            "            shiftgate adapter add <id> --runtime my-lora[/dim]"
        )
        return

    table = Table(
        title="Registered Adapters",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("ID", style="bold magenta", no_wrap=True)
    table.add_column("Name")
    table.add_column("Base Model", style="dim")
    table.add_column("Tags", style="green")
    table.add_column("Source", style="blue")
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="right")

    for a in adapters:
        source = _adapter_source_label(a)
        score = f"{a.benchmark_score:.2f}" if a.benchmark_score is not None else "—"
        tags = ", ".join(a.task_tags) if a.task_tags else "—"
        if a.status == "linked":
            status = "[green]linked[/green]"
        else:
            status = "[yellow]unassigned[/yellow]"
        table.add_row(a.id, a.name, a.base_model, tags, source, status, score)

    console.print(table)


# ---------------------------------------------------------------------------
# Task cluster table
# ---------------------------------------------------------------------------

def show_task_table(tasks: list[TaskCluster]) -> None:
    """Print a Rich table listing all task clusters and their preferred adapters."""
    if not tasks:
        console.print("[dim]No task clusters found.[/dim]")
        return

    table = Table(
        title="Task Clusters",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("ID", style="bold yellow", no_wrap=True)
    table.add_column("Name")
    table.add_column("Description", max_width=40)
    table.add_column("Preferred Adapters", style="magenta")
    table.add_column("Centroid", justify="center")

    for t in tasks:
        preferred = ", ".join(t.preferred_adapters) if t.preferred_adapters else "—"
        has_centroid = "[green]✓[/green]" if t.embedding_centroid else "[red]✗[/red]"
        table.add_row(t.id, t.name, t.description, preferred, has_centroid)

    console.print(table)


# ---------------------------------------------------------------------------
# Adapter swap animation
# ---------------------------------------------------------------------------

def animate_swap(from_adapter: str, to_adapter: str, duration: float = 1.5) -> None:
    """Show a short spinner animation while "swapping" from one adapter to another."""
    spinner = Spinner("dots", style="cyan")
    label = Text()
    label.append("Swapping  ", style="dim")
    label.append(from_adapter, style="bold yellow")
    label.append("  →  ", style="dim")
    label.append(to_adapter, style="bold green")

    with Live(Align.center(label), refresh_per_second=12, console=console) as live:
        end = time.monotonic() + duration
        while time.monotonic() < end:
            frame = spinner.render(time.monotonic())
            display = Text.assemble(frame, "  ", label)
            live.update(Align.center(display))
            time.sleep(0.08)

    console.print(
        f"  [green]✓[/green]  Swapped [yellow]{from_adapter}[/yellow] → [green]{to_adapter}[/green]"
    )


# ---------------------------------------------------------------------------
# Welcome / init banner
# ---------------------------------------------------------------------------

def show_welcome_banner() -> None:
    """Print a one-line banner (``shiftgate demo`` only)."""
    console.print("\n[bold cyan]⚡ shiftgate[/bold cyan]\n")


# ---------------------------------------------------------------------------
# Adapter acceptance / feedback stats table
# ---------------------------------------------------------------------------

def show_feedback_stats(scores: dict[str, float], stats: dict[str, int]) -> None:
    """Print a summary of adapter acceptance rates and overall trace stats."""
    console.print()
    console.print(
        f"[bold]Traces:[/bold]  "
        f"total=[cyan]{stats.get('total', 0)}[/cyan]  "
        f"accepted=[green]{stats.get('accepted', 0)}[/green]  "
        f"rejected=[red]{stats.get('rejected', 0)}[/red]  "
        f"unrated=[dim]{stats.get('unrated', 0)}[/dim]"
    )
    console.print()

    if not scores:
        console.print("[dim]No rated traces yet. Run `shiftgate feedback accept/reject` after routing.[/dim]")
        return

    table = Table(
        title="Adapter Acceptance Rates",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("Adapter ID", style="bold magenta")
    table.add_column("Acceptance Rate", justify="right")
    table.add_column("Bar")

    for adapter_id, rate in sorted(scores.items(), key=lambda x: -x[1]):
        rate_text = f"{rate * 100:.1f}%"
        bar = _similarity_bar(rate, width=15)
        table.add_row(adapter_id, rate_text, bar)

    console.print(table)


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def show_status(
    backend_name: str | None,
    n_adapters: int,
    n_tasks: int,
    embeddings_ready: bool,
) -> None:
    """Print a compact status summary for `shiftgate status`."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", min_width=20)
    grid.add_column()

    backend_style = "green" if backend_name else "red"
    backend_label = backend_name or "none detected"
    grid.add_row("Backend", Text(backend_label, style=f"bold {backend_style}"))
    grid.add_row("Adapters registered", Text(str(n_adapters), style="bold cyan"))
    grid.add_row("Task clusters", Text(str(n_tasks), style="bold cyan"))
    emb_style = "green" if embeddings_ready else "yellow"
    emb_label = "ready" if embeddings_ready else "not initialised — run `shiftgate init`"
    grid.add_row("Embeddings", Text(emb_label, style=emb_style))

    console.print(Panel(grid, title="shiftgate status", border_style="cyan", expand=False))


# ---------------------------------------------------------------------------
# Doctor report
# ---------------------------------------------------------------------------

def show_doctor_report(
    *,
    embedder_ok: bool,
    embedder_detail: str,
    backend_name: str | None,
    backend_url: str | None,
    adapter_rows: list[dict],
    n_tasks: int,
    n_with_embeddings: int,
    unlinked_tasks: list[str],
) -> None:
    """Render the full ``shiftgate doctor`` health report.

    Parameters mirror the checks performed in ``cli.doctor``.  Each section is
    a Rich panel/table; a final summary line tallies pass / warn / fail.
    """
    ok_mark = "[green]✓[/green]"
    warn_mark = "[yellow]⚠[/yellow]"
    fail_mark = "[red]✗[/red]"

    warnings = 0
    failures = 0

    console.print()
    console.rule("[bold cyan]shiftgate doctor[/bold cyan]")
    console.print()

    # --- Core checks grid ---
    core = Table.grid(padding=(0, 2))
    core.add_column(width=3)
    core.add_column(style="bold", min_width=18)
    core.add_column()

    # Embedder
    if embedder_ok:
        core.add_row(ok_mark, "Embedder", Text(f"loaded ({embedder_detail})", style="green"))
    else:
        failures += 1
        core.add_row(fail_mark, "Embedder", Text(f"failed: {embedder_detail}", style="red"))

    # Backend
    if backend_name:
        core.add_row(
            ok_mark,
            "Backend",
            Text(f"{backend_name}  ({backend_url})", style="green"),
        )
    else:
        warnings += 1
        core.add_row(
            warn_mark,
            "Backend",
            Text("none detected — start ollama serve or vLLM", style="yellow"),
        )

    # Task embeddings
    if n_tasks > 0 and n_with_embeddings == n_tasks:
        core.add_row(
            ok_mark,
            "Task embeddings",
            Text(f"{n_with_embeddings}/{n_tasks} clusters ready", style="green"),
        )
    else:
        warnings += 1
        core.add_row(
            warn_mark,
            "Task embeddings",
            Text(
                f"{n_with_embeddings}/{n_tasks} computed — run `shiftgate init`",
                style="yellow",
            ),
        )

    console.print(Panel(core, title="Core", border_style="cyan", expand=False))
    console.print()

    # --- Adapter availability table ---
    if adapter_rows:
        table = Table(
            title="Adapter runtime availability",
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="cyan",
        )
        table.add_column("Adapter ID", style="bold magenta")
        table.add_column("Backend name")
        table.add_column("Linked", justify="center")
        table.add_column("Loaded", justify="center")

        for row in adapter_rows:
            linked = (
                "[green]linked[/green]" if row["status"] == "linked"
                else "[yellow]unassigned[/yellow]"
            )
            state = row["state"]
            if state == "loaded":
                loaded = f"{ok_mark} loaded"
            elif state == "missing":
                warnings += 1
                loaded = f"{warn_mark} not loaded"
            else:  # unknown — no backend
                loaded = "[dim]— (no backend)[/dim]"
            table.add_row(row["id"], row["runtime"], linked, loaded)

        console.print(table)
    else:
        console.print("[dim]No adapters registered. Add one with `shiftgate adapter add`.[/dim]")
    console.print()

    # --- Unlinked task clusters warning ---
    if unlinked_tasks:
        warnings += 1
        console.print(
            Panel(
                Text(
                    "These task clusters have no linked adapter and will return "
                    "'No adapter available' if matched:\n  "
                    + ", ".join(unlinked_tasks),
                    style="yellow",
                ),
                title=f"{warn_mark} Unlinked task clusters ({len(unlinked_tasks)})",
                border_style="yellow",
                expand=False,
            )
        )
        console.print()

    # --- Summary line ---
    if failures:
        summary = f"[bold red]{failures} failed[/bold red]"
        if warnings:
            summary += f", [yellow]{warnings} warning(s)[/yellow]"
    elif warnings:
        summary = f"[bold yellow]{warnings} warning(s)[/bold yellow] — shiftgate is usable but check above"
    else:
        summary = "[bold green]All checks passed — shiftgate is healthy.[/bold green]"

    console.print(f"  {summary}")
    console.print()
