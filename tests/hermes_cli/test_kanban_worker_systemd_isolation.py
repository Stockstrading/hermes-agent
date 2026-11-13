"""Tests for systemd-run worker resource isolation wrapper.

Verifies that dispatcher-spawned workers are wrapped with systemd resource
limits (MemoryMax, CPUQuota) when systemd-run is available.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hermes_cli.kanban_db_dispatch import (
    _wrap_worker_with_systemd_scope,
    _get_worker_resource_config,
    WORKER_MEMORY_DEFAULT_MB,
    WORKER_CPU_QUOTA_DEFAULT,
)


class TestGetWorkerResourceConfig:
    """Test config loading for worker memory and CPU limits."""

    def test_defaults_when_config_empty(self):
        """Returns default limits when config is absent or empty."""
        with patch("hermes_cli.kanban_db_dispatch.load_config_readonly", return_value=None):
            mem_mb, cpu_quota = _get_worker_resource_config()
            assert mem_mb == WORKER_MEMORY_DEFAULT_MB
            assert cpu_quota == WORKER_CPU_QUOTA_DEFAULT

    def test_defaults_when_kanban_section_missing(self):
        """Returns defaults when kanban section not in config."""
        with patch("hermes_cli.kanban_db_dispatch.load_config_readonly", return_value={}):
            mem_mb, cpu_quota = _get_worker_resource_config()
            assert mem_mb == WORKER_MEMORY_DEFAULT_MB
            assert cpu_quota == WORKER_CPU_QUOTA_DEFAULT

    def test_reads_custom_memory_limit(self):
        """Reads custom worker_memory_mb from config."""
        with patch(
            "hermes_cli.kanban_db_dispatch.load_config_readonly",
            return_value={"kanban": {"worker_memory_mb": 2048}},
        ):
            mem_mb, cpu_quota = _get_worker_resource_config()
            assert mem_mb == 2048
            assert cpu_quota == WORKER_CPU_QUOTA_DEFAULT

    def test_reads_custom_cpu_quota(self):
        """Reads custom worker_cpu_quota from config."""
        with patch(
            "hermes_cli.kanban_db_dispatch.load_config_readonly",
            return_value={"kanban": {"worker_cpu_quota": "200%"}},
        ):
            mem_mb, cpu_quota = _get_worker_resource_config()
            assert mem_mb == WORKER_MEMORY_DEFAULT_MB
            assert cpu_quota == "200%"

    def test_clamps_invalid_memory(self):
        """Rejects invalid memory values and falls back to default."""
        test_cases = [
            ({"kanban": {"worker_memory_mb": 32}}, WORKER_MEMORY_DEFAULT_MB),  # too low
            ({"kanban": {"worker_memory_mb": "1024"}}, WORKER_MEMORY_DEFAULT_MB),  # string, not int
            ({"kanban": {"worker_memory_mb": -512}}, WORKER_MEMORY_DEFAULT_MB),  # negative
        ]
        for config, expected_mem in test_cases:
            with patch("hermes_cli.kanban_db_dispatch.load_config_readonly", return_value=config):
                mem_mb, _ = _get_worker_resource_config()
                assert mem_mb == expected_mem

    def test_config_load_exception_fallback(self):
        """Falls back to defaults on config load exceptions."""
        with patch(
            "hermes_cli.kanban_db_dispatch.load_config_readonly",
            side_effect=Exception("config read failed"),
        ):
            mem_mb, cpu_quota = _get_worker_resource_config()
            assert mem_mb == WORKER_MEMORY_DEFAULT_MB
            assert cpu_quota == WORKER_CPU_QUOTA_DEFAULT


@pytest.mark.platforms("posix")  # systemd-run is Unix only
class TestWrapWorkerWithSystemdScope:
    """Test worker command wrapping with systemd-run."""

    def test_no_wrap_on_windows(self):
        """On Windows, returns command unchanged."""
        task = Mock(id="test_task")
        cmd = ["hermes", "-p", "default", "chat", "-q"]
        
        with patch("hermes_cli.kanban_db_dispatch._kb._IS_WINDOWS", True):
            result = _wrap_worker_with_systemd_scope(task, cmd)
            assert result == cmd

    def test_no_wrap_when_systemd_run_unavailable(self):
        """When systemd-run not on PATH, returns command unchanged."""
        task = Mock(id="test_task")
        cmd = ["hermes", "-p", "default", "chat", "-q"]
        
        with patch("hermes_cli.kanban_db_dispatch._kb._IS_WINDOWS", False):
            with patch("shutil.which", return_value=None):
                result = _wrap_worker_with_systemd_scope(task, cmd)
                assert result == cmd

    def test_wraps_with_systemd_run_when_available(self):
        """When systemd-run is available, wraps command with resource limits."""
        task = Mock(id="test_task")
        cmd = ["hermes", "-p", "default", "chat", "-q"]
        
        with patch("hermes_cli.kanban_db_dispatch._kb._IS_WINDOWS", False):
            with patch("shutil.which", return_value="/bin/systemd-run"):
                with patch(
                    "hermes_cli.kanban_db_dispatch._get_worker_resource_config",
                    return_value=(1024, "100%"),
                ):
                    result = _wrap_worker_with_systemd_scope(task, cmd)
        
        # Expect: systemd-run --user --scope -p MemoryMax=1024M -p CPUQuota=100% -- <cmd>
        assert result[0] == "systemd-run"
        assert result[1] == "--user"
        assert result[2] == "--scope"
        assert "-p" in result
        assert "MemoryMax=1024M" in result
        assert "CPUQuota=100%" in result
        assert "--" in result
        # Original command should be at the end, after --
        split_idx = result.index("--")
        assert result[split_idx + 1:] == cmd

    def test_uses_custom_limits(self):
        """Uses custom memory and CPU limits from config."""
        task = Mock(id="test_task")
        cmd = ["python", "script.py"]
        
        with patch("hermes_cli.kanban_db_dispatch._kb._IS_WINDOWS", False):
            with patch("shutil.which", return_value="/bin/systemd-run"):
                with patch(
                    "hermes_cli.kanban_db_dispatch._get_worker_resource_config",
                    return_value=(2048, "200%"),
                ):
                    result = _wrap_worker_with_systemd_scope(task, cmd)
        
        assert "MemoryMax=2048M" in result
        assert "CPUQuota=200%" in result

    def test_preserves_original_command_order(self):
        """Original command is preserved exactly after -- separator."""
        task = Mock(id="test_task")
        cmd = ["hermes", "-p", "my-profile", "chat", "-q", "arg1", "arg2"]
        
        with patch("hermes_cli.kanban_db_dispatch._kb._IS_WINDOWS", False):
            with patch("shutil.which", return_value="/bin/systemd-run"):
                with patch(
                    "hermes_cli.kanban_db_dispatch._get_worker_resource_config",
                    return_value=(1024, "100%"),
                ):
                    result = _wrap_worker_with_systemd_scope(task, cmd)
        
        split_idx = result.index("--")
        assert result[split_idx + 1:] == cmd
