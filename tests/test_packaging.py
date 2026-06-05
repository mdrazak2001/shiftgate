"""
Packaging tests — guard against the 0.1.3 regression where the published wheel
shipped without its Python modules (``ModuleNotFoundError: shiftgate.cli``).

Why these tests build the *full* pipeline
-----------------------------------------
PyPI (and ``uv build`` / ``python -m build`` with no flags) builds the sdist
first, then builds the wheel **from the extracted sdist**.  A wheel built
directly from the source tree can look complete while the real published wheel
(built from a too-restrictive sdist) is missing modules.  So the regression
guard below uses the default ``uv build`` (sdist + wheel-from-sdist), exactly
like a release, and inspects the resulting wheel.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Skip the whole module gracefully if `uv` isn't on PATH (e.g. minimal CI image).
_UV = shutil.which("uv")
pytestmark = pytest.mark.skipif(_UV is None, reason="uv is required to build the package")


def _run_build(*args: str, out_dir: Path) -> None:
    """Build into ``out_dir``. With no extra args, builds sdist then wheel-from-sdist."""
    subprocess.run(
        [_UV, "build", *args, "-o", str(out_dir)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def full_dist(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build the sdist and a wheel-from-sdist once, like a real release.

    Returns a dict with ``wheel`` and ``sdist`` paths.
    """
    dist_dir = tmp_path_factory.mktemp("dist")
    _run_build(out_dir=dist_dir)  # no flags → sdist + wheel built from that sdist

    wheels = list(dist_dir.glob("*.whl"))
    sdists = list(dist_dir.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"
    assert len(sdists) == 1, f"expected one sdist, got {sdists}"
    return {"wheel": wheels[0], "sdist": sdists[0]}


def _wheel_names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        return archive.namelist()


def _sdist_names(sdist: Path) -> list[str]:
    with tarfile.open(sdist, "r:gz") as archive:
        return archive.getnames()


# ---------------------------------------------------------------------------
# Wheel content tests
# ---------------------------------------------------------------------------

class TestWheelContents:
    def test_wheel_contains_cli_module(self, full_dist: dict[str, Path]) -> None:
        """The 0.1.3 regression guard: shiftgate/cli.py must be in the wheel."""
        names = _wheel_names(full_dist["wheel"])
        assert "shiftgate/cli.py" in names, (
            "shiftgate/cli.py missing from wheel — this is the 0.1.3 packaging bug. "
            f"Wheel contained: {names}"
        )

    def test_wheel_contains_all_subpackages(self, full_dist: dict[str, Path]) -> None:
        """Every shiftgate subpackage module should ship in the wheel."""
        names = set(_wheel_names(full_dist["wheel"]))
        expected = {
            "shiftgate/__init__.py",
            "shiftgate/cli.py",
            "shiftgate/registry/schemas.py",
            "shiftgate/registry/adapter_registry.py",
            "shiftgate/registry/task_registry.py",
            "shiftgate/router/router.py",
            "shiftgate/router/matcher.py",
            "shiftgate/router/embedder.py",
            "shiftgate/runtime/backend.py",
            "shiftgate/feedback/loop.py",
            "shiftgate/utils/display.py",
            "shiftgate/serve/__init__.py",
            "shiftgate/serve/app.py",
        }
        missing = expected - names
        assert not missing, f"wheel is missing modules: {sorted(missing)}"

    def test_wheel_contains_default_tasks_json(self, full_dist: dict[str, Path]) -> None:
        names = _wheel_names(full_dist["wheel"])
        assert "shiftgate/data/default_tasks.json" in names

    def test_wheel_default_tasks_json_is_valid(self, full_dist: dict[str, Path]) -> None:
        with zipfile.ZipFile(full_dist["wheel"]) as archive:
            raw = archive.read("shiftgate/data/default_tasks.json")
        tasks = json.loads(raw)
        assert isinstance(tasks, list)
        assert len(tasks) == 10
        assert tasks[0]["id"]

    def test_wheel_is_not_truncated(self, full_dist: dict[str, Path]) -> None:
        """A healthy wheel is well above the ~8 KB broken build."""
        size_kb = full_dist["wheel"].stat().st_size / 1024
        assert size_kb > 15, f"wheel is suspiciously small ({size_kb:.1f} KB)"


# ---------------------------------------------------------------------------
# Sdist content tests
# ---------------------------------------------------------------------------

class TestSdistContents:
    def test_sdist_contains_cli_module(self, full_dist: dict[str, Path]) -> None:
        names = _sdist_names(full_dist["sdist"])
        assert any(n.endswith("shiftgate/cli.py") for n in names), (
            "shiftgate/cli.py missing from sdist — wheels built from it will be broken."
        )

    def test_sdist_contains_default_tasks_json(self, full_dist: dict[str, Path]) -> None:
        names = _sdist_names(full_dist["sdist"])
        assert any(n.endswith("shiftgate/data/default_tasks.json") for n in names)


# ---------------------------------------------------------------------------
# Integration test: install the wheel into a fresh venv and run the CLI
# ---------------------------------------------------------------------------

class TestWheelInstall:
    def test_installed_wheel_cli_help_runs(
        self, full_dist: dict[str, Path], tmp_path: Path
    ) -> None:
        """Install the wheel into a clean venv and run `shiftgate --help`.

        Verifies the console-script entry point resolves and ``shiftgate.cli``
        imports cleanly from the installed package (exit code 0).
        """
        venv_dir = tmp_path / "venv"
        subprocess.run([_UV, "venv", str(venv_dir)], check=True, capture_output=True, text=True)

        # Install the freshly built wheel (deps resolved from uv's cache).
        subprocess.run(
            [_UV, "pip", "install", "--python", str(venv_dir), str(full_dist["wheel"])],
            check=True,
            capture_output=True,
            text=True,
        )

        # Locate the installed console script (cross-platform).
        if os.name == "nt":
            script = venv_dir / "Scripts" / "shiftgate.exe"
        else:
            script = venv_dir / "bin" / "shiftgate"
        assert script.exists(), f"console script not installed at {script}"

        # Force UTF-8 decoding: Rich help output contains Unicode glyphs that
        # crash the default cp1252 reader on Windows.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [str(script), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert result.returncode == 0, (
            f"`shiftgate --help` failed (rc={result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "shiftgate" in result.stdout.lower()
