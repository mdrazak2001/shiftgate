"""
Packaging tests: verify bundled data ships in wheel and sdist builds.
"""

from __future__ import annotations

import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_build(*args: str, out_dir: Path) -> None:
    subprocess.run(
        ["uv", "build", *args, "-o", str(out_dir)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


class TestPackaging:
    def test_wheel_contains_default_tasks_json(self, tmp_path: Path) -> None:
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        _run_build("--wheel", out_dir=dist_dir)

        wheels = list(dist_dir.glob("*.whl"))
        assert len(wheels) == 1

        with zipfile.ZipFile(wheels[0]) as archive:
            names = archive.namelist()

        assert "shiftgate/data/default_tasks.json" in names

    def test_wheel_default_tasks_json_is_valid(self, tmp_path: Path) -> None:
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        _run_build("--wheel", out_dir=dist_dir)
        wheel = next(dist_dir.glob("*.whl"))

        with zipfile.ZipFile(wheel) as archive:
            raw = archive.read("shiftgate/data/default_tasks.json")

        tasks = json.loads(raw)
        assert isinstance(tasks, list)
        assert len(tasks) == 10
        assert tasks[0]["id"]

    def test_sdist_contains_default_tasks_json(self, tmp_path: Path) -> None:
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        _run_build("--sdist", out_dir=dist_dir)

        sdists = list(dist_dir.glob("*.tar.gz"))
        assert len(sdists) == 1

        with tarfile.open(sdists[0], "r:gz") as archive:
            members = archive.getnames()

        assert any(member.endswith("shiftgate/data/default_tasks.json") for member in members)
