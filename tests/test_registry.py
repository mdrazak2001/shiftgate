"""
Tests for the adapter and task registries.

Covers: add/get/list, save/load round-trip, and duplicate handling.
All file I/O is redirected to a temporary directory via monkeypatching so
tests never pollute ~/.shiftgate/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shiftgate.registry.adapter_registry import AdapterRegistry
from shiftgate.registry.schemas import AdapterEntry, TaskCluster
from shiftgate.registry.task_registry import TaskRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_shiftgate(tmp_path, monkeypatch):
    """Redirect all registry file I/O to a temporary directory."""
    import shiftgate.registry.adapter_registry as ar_mod
    import shiftgate.registry.task_registry as tr_mod

    shiftgate_dir = tmp_path / ".shiftgate"
    shiftgate_dir.mkdir()

    monkeypatch.setattr(ar_mod, "_SHIFTGATE_DIR", shiftgate_dir)
    monkeypatch.setattr(ar_mod, "_USER_ADAPTERS_PATH", shiftgate_dir / "adapters.json")
    monkeypatch.setattr(tr_mod, "_SHIFTGATE_DIR", shiftgate_dir)
    monkeypatch.setattr(tr_mod, "_USER_TASKS_PATH", shiftgate_dir / "tasks.json")
    monkeypatch.setattr(tr_mod, "_CACHE_PATH", shiftgate_dir / "embeddings_cache.npy")

    return shiftgate_dir


@pytest.fixture()
def sample_adapter() -> AdapterEntry:
    return AdapterEntry(
        id="test-lora",
        name="Test LoRA",
        base_model="meta-llama/Meta-Llama-3-8B",
        task_tags=["code", "python"],
        description="A test adapter",
        hf_repo="test-user/test-lora",
    )


@pytest.fixture()
def sample_task() -> TaskCluster:
    return TaskCluster(
        id="test_task",
        name="Test Task",
        description="A task for testing",
        validation_examples=[
            "write a test",
            "create a unit test",
            "generate a pytest fixture",
        ],
        preferred_adapters=["test-lora"],
    )


# ---------------------------------------------------------------------------
# AdapterRegistry tests
# ---------------------------------------------------------------------------

class TestAdapterRegistry:
    def test_empty_registry(self, tmp_shiftgate):
        """Loading with no file yields an empty registry."""
        reg = AdapterRegistry(adapters=[], source_path=tmp_shiftgate / "adapters.json")
        assert len(reg) == 0
        assert reg.list_adapters() == []

    def test_add_and_get(self, tmp_shiftgate, sample_adapter):
        reg = AdapterRegistry(adapters=[], source_path=tmp_shiftgate / "adapters.json")
        reg.add_adapter(sample_adapter)
        assert len(reg) == 1
        result = reg.get_adapter("test-lora")
        assert result is not None
        assert result.id == "test-lora"
        assert result.base_model == "meta-llama/Meta-Llama-3-8B"

    def test_get_missing_returns_none(self, tmp_shiftgate):
        reg = AdapterRegistry(adapters=[], source_path=tmp_shiftgate / "adapters.json")
        assert reg.get_adapter("does-not-exist") is None

    def test_list_adapters(self, tmp_shiftgate, sample_adapter):
        reg = AdapterRegistry(adapters=[sample_adapter], source_path=tmp_shiftgate / "adapters.json")
        adapters = reg.list_adapters()
        assert len(adapters) == 1
        assert adapters[0].id == "test-lora"

    def test_remove_adapter(self, tmp_shiftgate, sample_adapter):
        reg = AdapterRegistry(adapters=[sample_adapter], source_path=tmp_shiftgate / "adapters.json")
        removed = reg.remove_adapter("test-lora")
        assert removed is True
        assert len(reg) == 0

    def test_remove_nonexistent_returns_false(self, tmp_shiftgate):
        reg = AdapterRegistry(adapters=[], source_path=tmp_shiftgate / "adapters.json")
        assert reg.remove_adapter("phantom") is False

    def test_overwrite_existing_adapter(self, tmp_shiftgate, sample_adapter):
        reg = AdapterRegistry(adapters=[sample_adapter], source_path=tmp_shiftgate / "adapters.json")
        updated = AdapterEntry(
            id="test-lora",
            name="Updated LoRA",
            base_model="mistralai/Mistral-7B",
            task_tags=[],
        )
        reg.add_adapter(updated)
        assert len(reg) == 1
        assert reg.get_adapter("test-lora").name == "Updated LoRA"

    def test_save_and_load_round_trip(self, tmp_shiftgate, sample_adapter, monkeypatch):
        """Save registry to tmp dir, then reload and assert fidelity."""
        import shiftgate.registry.adapter_registry as ar_mod

        save_path = tmp_shiftgate / "adapters.json"
        monkeypatch.setattr(ar_mod, "_USER_ADAPTERS_PATH", save_path)

        reg = AdapterRegistry(adapters=[sample_adapter], source_path=save_path)
        reg.save()

        assert save_path.exists()
        raw = json.loads(save_path.read_text())
        assert len(raw) == 1
        assert raw[0]["id"] == "test-lora"

        reg2 = AdapterRegistry.load()
        assert len(reg2) == 1
        loaded = reg2.get_adapter("test-lora")
        assert loaded.hf_repo == "test-user/test-lora"
        assert loaded.task_tags == ["code", "python"]


# ---------------------------------------------------------------------------
# TaskRegistry tests
# ---------------------------------------------------------------------------

class TestTaskRegistry:
    def test_add_and_get_task(self, tmp_shiftgate, sample_task):
        reg = TaskRegistry(tasks=[], source_path=tmp_shiftgate / "tasks.json")
        reg.add_task(sample_task)
        assert len(reg) == 1
        result = reg.get_task("test_task")
        assert result is not None
        assert result.name == "Test Task"

    def test_get_missing_task_returns_none(self, tmp_shiftgate):
        reg = TaskRegistry(tasks=[], source_path=tmp_shiftgate / "tasks.json")
        assert reg.get_task("nonexistent") is None

    def test_get_all_tasks(self, tmp_shiftgate, sample_task):
        reg = TaskRegistry(tasks=[sample_task], source_path=tmp_shiftgate / "tasks.json")
        all_tasks = reg.get_all_tasks()
        assert len(all_tasks) == 1
        assert all_tasks[0].id == "test_task"

    def test_remove_task(self, tmp_shiftgate, sample_task):
        reg = TaskRegistry(tasks=[sample_task], source_path=tmp_shiftgate / "tasks.json")
        removed = reg.remove_task("test_task")
        assert removed is True
        assert len(reg) == 0

    def test_remove_nonexistent_task(self, tmp_shiftgate):
        reg = TaskRegistry(tasks=[], source_path=tmp_shiftgate / "tasks.json")
        assert reg.remove_task("phantom") is False

    def test_embeddings_not_ready_without_centroid(self, tmp_shiftgate, sample_task):
        """A task without a centroid means embeddings are not ready."""
        assert sample_task.embedding_centroid is None
        reg = TaskRegistry(tasks=[sample_task], source_path=tmp_shiftgate / "tasks.json")
        assert reg.embeddings_ready() is False

    def test_embeddings_ready_with_centroid(self, tmp_shiftgate, sample_task):
        sample_task.embedding_centroid = [0.1, 0.2, 0.3]
        reg = TaskRegistry(tasks=[sample_task], source_path=tmp_shiftgate / "tasks.json")
        assert reg.embeddings_ready() is True

    def test_save_and_load_round_trip(self, tmp_shiftgate, sample_task, monkeypatch):
        import shiftgate.registry.task_registry as tr_mod

        save_path = tmp_shiftgate / "tasks.json"
        monkeypatch.setattr(tr_mod, "_USER_TASKS_PATH", save_path)

        reg = TaskRegistry(tasks=[sample_task], source_path=save_path)
        reg.save()

        assert save_path.exists()
        raw = json.loads(save_path.read_text())
        assert len(raw) == 1
        assert raw[0]["id"] == "test_task"

        reg2 = TaskRegistry.load()
        loaded = reg2.get_task("test_task")
        assert loaded is not None
        assert loaded.preferred_adapters == ["test-lora"]
        assert loaded.validation_examples[0] == "write a test"

    def test_add_task_overwrites_existing(self, tmp_shiftgate, sample_task):
        reg = TaskRegistry(tasks=[sample_task], source_path=tmp_shiftgate / "tasks.json")
        updated = TaskCluster(
            id="test_task",
            name="Updated Task",
            description="Changed description",
            validation_examples=["new example"],
        )
        reg.add_task(updated)
        assert len(reg) == 1
        assert reg.get_task("test_task").name == "Updated Task"


# ---------------------------------------------------------------------------
# _auto_link_adapter helper tests
# ---------------------------------------------------------------------------

class TestAutoLinkAdapter:
    """Tests for the cli._auto_link_adapter helper."""

    def _make_task_reg(self, task_ids: list[str], tmp_path) -> TaskRegistry:
        tasks = [
            TaskCluster(
                id=tid,
                name=tid,
                description="",
                validation_examples=["example"],
                preferred_adapters=[],
            )
            for tid in task_ids
        ]
        return TaskRegistry(tasks=tasks, source_path=tmp_path / "tasks.json")

    def test_links_matching_task(self, tmp_path):
        from shiftgate.cli import _auto_link_adapter

        task_reg = self._make_task_reg(["code_sql", "code_python"], tmp_path)
        adapter = AdapterEntry(id="sql-lora", name="SQL", base_model="x", task_tags=["sql"])

        linked = _auto_link_adapter(adapter, task_reg)

        assert "code_sql" in linked
        assert "code_python" not in linked
        assert "sql-lora" in task_reg.get_task("code_sql").preferred_adapters
        assert "sql-lora" not in task_reg.get_task("code_python").preferred_adapters

    def test_links_multiple_tasks(self, tmp_path):
        from shiftgate.cli import _auto_link_adapter

        task_reg = self._make_task_reg(["code_sql", "code_python", "text_summarize"], tmp_path)
        adapter = AdapterEntry(id="code-lora", name="Code", base_model="x", task_tags=["code"])

        linked = _auto_link_adapter(adapter, task_reg)

        # "code" matches "code_sql" and "code_python" but not "text_summarize"
        assert set(linked) == {"code_sql", "code_python"}

    def test_no_duplicate_links(self, tmp_path):
        from shiftgate.cli import _auto_link_adapter

        task_reg = self._make_task_reg(["code_sql"], tmp_path)
        task_reg.get_task("code_sql").preferred_adapters = ["sql-lora"]  # already linked
        adapter = AdapterEntry(id="sql-lora", name="SQL", base_model="x", task_tags=["sql"])

        linked = _auto_link_adapter(adapter, task_reg)

        assert linked == []
        assert task_reg.get_task("code_sql").preferred_adapters == ["sql-lora"]

    def test_no_tags_returns_empty(self, tmp_path):
        from shiftgate.cli import _auto_link_adapter

        task_reg = self._make_task_reg(["code_sql"], tmp_path)
        adapter = AdapterEntry(id="unlabelled", name="X", base_model="x", task_tags=[])

        linked = _auto_link_adapter(adapter, task_reg)
        assert linked == []
