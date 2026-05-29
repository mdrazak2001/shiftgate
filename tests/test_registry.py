"""
Tests for the adapter and task registries.

Covers: add/get/list, save/load round-trip, duplicate handling,
three registration modes, and the _auto_link_adapter helper.
All file I/O is redirected to a temporary directory via monkeypatching so
tests never pollute ~/.shiftgate/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shiftgate.registry.adapter_registry import (
    AdapterRegistry,
    adapter_from_hf,
    adapter_from_local,
    adapter_from_runtime,
)
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
    def test_empty_registry_when_no_file(self, tmp_shiftgate):
        """Loading with no file yields an empty registry (no error)."""
        reg = AdapterRegistry.load()
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

    def test_backward_compat_old_entries_without_runtime_name(self, tmp_shiftgate, monkeypatch):
        """Old adapters.json files without runtime_name should load cleanly."""
        import shiftgate.registry.adapter_registry as ar_mod

        save_path = tmp_shiftgate / "adapters.json"
        monkeypatch.setattr(ar_mod, "_USER_ADAPTERS_PATH", save_path)

        # Write a legacy entry that lacks runtime_name and other new fields.
        legacy_data = [
            {
                "id": "old-lora",
                "name": "Old LoRA",
                "base_model": "llama2",
                "task_tags": ["code"],
                "description": "Legacy entry",
                "hf_repo": "org/old-lora",
                "local_path": None,
                "benchmark_score": None,
                "context_length": 4096,   # field removed in v0.1 BYOM; should be ignored
                "memory_mb": None,         # same
            }
        ]
        save_path.write_text(json.dumps(legacy_data))

        reg = AdapterRegistry.load()
        assert len(reg) == 1
        adapter = reg.get_adapter("old-lora")
        assert adapter is not None
        assert adapter.runtime_name is None   # new field defaults correctly


# ---------------------------------------------------------------------------
# Registration-mode factory tests
# ---------------------------------------------------------------------------

class TestRegistrationModes:
    def test_adapter_from_hf_basic(self):
        adapter = adapter_from_hf(
            "teknium/sql-lora",
            tags=["sql"],
            base_model="llama3",
        )
        assert adapter.id == "sql-lora"
        assert adapter.hf_repo == "teknium/sql-lora"
        assert adapter.local_path is None
        assert adapter.runtime_name is None
        assert adapter.task_tags == ["sql"]
        assert adapter.base_model == "llama3"

    def test_adapter_from_hf_custom_id(self):
        adapter = adapter_from_hf("org/my-adapter", adapter_id="custom-id")
        assert adapter.id == "custom-id"

    def test_adapter_from_hf_slug_derivation(self):
        """Slug is the last path component, lowercased, underscores→hyphens."""
        adapter = adapter_from_hf("org/My_Adapter_v2")
        assert adapter.id == "my-adapter-v2"

    def test_adapter_from_local(self):
        adapter = adapter_from_local(
            local_path="/models/sql-lora",
            adapter_id="sql-lora",
            tags=["sql"],
            base_model="llama3",
        )
        assert adapter.id == "sql-lora"
        assert adapter.local_path == "/models/sql-lora"
        assert adapter.hf_repo is None
        assert adapter.runtime_name is None
        assert adapter.task_tags == ["sql"]

    def test_adapter_from_runtime(self):
        adapter = adapter_from_runtime(
            runtime_name="sql-lora-vllm",
            adapter_id="sql-lora",
            tags=["sql"],
            base_model="llama3",
        )
        assert adapter.id == "sql-lora"
        assert adapter.runtime_name == "sql-lora-vllm"
        assert adapter.hf_repo is None
        assert adapter.local_path is None

    def test_adapter_from_runtime_slug_default_id(self):
        """When adapter_id omitted, slug is derived from runtime_name."""
        adapter = adapter_from_runtime("sql-lora-vllm")
        assert adapter.id == "sql-lora-vllm"
        assert adapter.runtime_name == "sql-lora-vllm"

    def test_effective_backend_name_prefers_runtime_name(self):
        adapter = adapter_from_runtime("vllm-name", adapter_id="my-lora")
        assert adapter.effective_backend_name() == "vllm-name"

    def test_effective_backend_name_falls_back_to_id(self):
        adapter = adapter_from_hf("org/my-lora", adapter_id="my-lora")
        assert adapter.effective_backend_name() == "my-lora"


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

    def test_load_bundled_defaults_when_no_user_registry(self, tmp_shiftgate):
        """Packaged defaults load via importlib when ~/.shiftgate/tasks.json is absent."""
        reg = TaskRegistry.load()
        assert len(reg) == 10
        assert reg.get_task("code_python") is not None
        assert reg.get_task("code_sql") is not None

    def test_user_registry_takes_priority_over_bundled_defaults(self, tmp_shiftgate, sample_task):
        """Existing user registries continue to win over bundled defaults."""
        import shiftgate.registry.task_registry as tr_mod

        user_path = tmp_shiftgate / "tasks.json"
        user_path.write_text(
            json.dumps([sample_task.model_dump()]),
            encoding="utf-8",
        )
        assert tr_mod._USER_TASKS_PATH == user_path

        reg = TaskRegistry.load()
        assert len(reg) == 1
        assert reg.get_task("test_task") is not None
        assert reg.get_task("code_python") is None


# ---------------------------------------------------------------------------
# _auto_link_adapter helper tests
# ---------------------------------------------------------------------------

class TestAutoLinkAdapter:
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

        assert set(linked) == {"code_sql", "code_python"}

    def test_no_duplicate_links(self, tmp_path):
        from shiftgate.cli import _auto_link_adapter

        task_reg = self._make_task_reg(["code_sql"], tmp_path)
        task_reg.get_task("code_sql").preferred_adapters = ["sql-lora"]
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


# ---------------------------------------------------------------------------
# Adapter status field
# ---------------------------------------------------------------------------

class TestAdapterStatus:
    def test_default_status_is_unassigned(self):
        adapter = AdapterEntry(id="x", name="X", base_model="b")
        assert adapter.status == "unassigned"

    def test_status_persists_round_trip(self, tmp_shiftgate, monkeypatch):
        import shiftgate.registry.adapter_registry as ar_mod

        save_path = tmp_shiftgate / "adapters.json"
        monkeypatch.setattr(ar_mod, "_USER_ADAPTERS_PATH", save_path)

        adapter = AdapterEntry(
            id="sql-lora", name="SQL", base_model="llama3",
            task_tags=["sql"], hf_repo="org/sql-lora", status="linked",
        )
        AdapterRegistry(adapters=[adapter], source_path=save_path).save()

        reloaded = AdapterRegistry.load().get_adapter("sql-lora")
        assert reloaded.status == "linked"

    def test_finish_adapter_add_marks_linked(self, tmp_shiftgate, monkeypatch):
        """_finish_adapter_add should set status='linked' when a task is linked."""
        import shiftgate.cli as cli_mod

        # Quiet console output during the test.
        monkeypatch.setattr(cli_mod.console, "print", lambda *a, **k: None)

        task = TaskCluster(
            id="code_sql", name="SQL", description="",
            validation_examples=["x"], preferred_adapters=[],
        )
        task_reg = TaskRegistry(tasks=[task], source_path=tmp_shiftgate / "tasks.json")
        adapter = AdapterEntry(
            id="sql-lora", name="SQL", base_model="llama3",
            task_tags=["sql"], hf_repo="org/sql-lora",
        )
        adapter_reg = AdapterRegistry(adapters=[adapter], source_path=tmp_shiftgate / "adapters.json")

        cli_mod._finish_adapter_add(adapter, task_reg, adapter_reg)
        assert adapter.status == "linked"

    def test_finish_adapter_add_stays_unassigned_without_match(self, tmp_shiftgate, monkeypatch):
        import shiftgate.cli as cli_mod

        monkeypatch.setattr(cli_mod.console, "print", lambda *a, **k: None)

        task = TaskCluster(
            id="code_sql", name="SQL", description="",
            validation_examples=["x"], preferred_adapters=[],
        )
        task_reg = TaskRegistry(tasks=[task], source_path=tmp_shiftgate / "tasks.json")
        adapter = AdapterEntry(
            id="music-lora", name="Music", base_model="llama3",
            task_tags=["music"], hf_repo="org/music-lora",
        )
        adapter_reg = AdapterRegistry(adapters=[adapter], source_path=tmp_shiftgate / "adapters.json")

        cli_mod._finish_adapter_add(adapter, task_reg, adapter_reg)
        assert adapter.status == "unassigned"
