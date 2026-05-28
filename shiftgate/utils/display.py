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


# ---------------------------------------------------------------------------
# Routing decision panel
# ---------------------------------------------------------------------------

def show_routing_decision(
    trace: RoutingTrace,
    adapter: AdapterEntry | None = None,
    task_name: str | None = None,
    backend_name: str | None = None,
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
        Active backend name ('ollama', 'vllm', or None).
    """
    colour = _similarity_colour(trace.similarity_score)

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

    if adapter:
        adapter_text = Text()
        adapter_text.append(adapter.name, style="bold magenta")
        adapter_text.append(f"  [{adapter.base_model}]", style="dim")
        if adapter.hf_repo:
            adapter_text.append(f"\n  hf: {adapter.hf_repo}", style="dim blue")
        grid.add_row("Adapter", adapter_text)
    else:
        grid.add_row("Adapter", Text(trace.selected_adapter_id, style="bold magenta"))

    backend_text = Text(backend_name or "—", style="green" if backend_name else "dim")
    grid.add_row("Backend", backend_text)

    if trace.latency_ms is not None:
        grid.add_row("Latency", Text(f"{trace.latency_ms:.0f} ms", style="dim"))

    panel = Panel(
        grid,
        title=f"[bold {colour}] shiftgate routing decision [/bold {colour}]",
        border_style=colour,
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()


# ---------------------------------------------------------------------------
# Adapter table
# ---------------------------------------------------------------------------

def show_adapter_table(adapters: list[AdapterEntry]) -> None:
    """Print a Rich table listing all registered adapters."""
    if not adapters:
        console.print("[dim]No adapters registered. Add one with `shiftgate adapter add <hf_repo>`.[/dim]")
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
    table.add_column("HF Repo / Local Path", style="blue")
    table.add_column("Score", justify="right")

    for a in adapters:
        location = a.hf_repo or a.local_path or "—"
        score = f"{a.benchmark_score:.2f}" if a.benchmark_score is not None else "—"
        tags = ", ".join(a.task_tags) if a.task_tags else "—"
        table.add_row(a.id, a.name, a.base_model, tags, location, score)

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
    """Print the shiftgate welcome banner shown during `shiftgate init`."""
    banner = Text(justify="center")
    banner.append("\n  ⚡ shiftgate  ", style="bold cyan")
    banner.append("v0.1\n", style="dim")
    banner.append("  Intelligent LoRA routing for local LLM inference\n", style="italic white")
    banner.append("  Inspired by LORAUTER · EPFL 2026\n\n", style="dim")

    panel = Panel(
        Align.center(banner),
        border_style="cyan",
        expand=False,
    )
    console.print(Align.center(panel))


# ---------------------------------------------------------------------------
# Adapter acceptance / feedback stats table
# ---------------------------------------------------------------------------

def show_feedback_stats(scores: dict[str, float], stats: dict[str, int]) -> None:
    """Print a summary of adapter acceptance rates and overall trace stats.

    Parameters
    ----------
    scores:
        Output of ``feedback.loop.compute_adapter_scores()``.
    stats:
        Output of ``feedback.loop.get_trace_stats()``.
    """
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
