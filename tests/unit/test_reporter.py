"""Tests for cython_framework.analysis.reporter.

TDD: Tests cover terminal, JSON, GitHub issue formatting and CLI behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cython_framework.analysis.reporter import (
    create_github_issue,
    format_github_issue,
    format_json,
    format_terminal,
)
from cython_framework.manifest import (
    AnnotateScore,
    Hotspot,
    Manifest,
    ModuleEntry,
    add_module,
    save,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    tier: int = 1,
    status: str = "unanalyzed",
    annotate_score: AnnotateScore | None = None,
    rust_candidate: bool = False,
    rust_rationale: str | None = None,
    github_issue: int | None = None,
) -> ModuleEntry:
    return ModuleEntry(
        tier=tier,
        source_type="pyx",
        source_path="module/path.pyx",
        status=status,
        annotate_score=annotate_score,
        rust_candidate=rust_candidate,
        rust_rationale=rust_rationale,
        github_issue=github_issue,
    )


def _make_manifest(*module_items: tuple[str, ModuleEntry]) -> Manifest:
    manifest = Manifest(project="test_project")
    for key, entry in module_items:
        manifest = add_module(manifest, key, entry)
    return manifest


# ---------------------------------------------------------------------------
# format_terminal
# ---------------------------------------------------------------------------


class TestFormatTerminal:
    def test_empty_manifest_returns_sentinel(self):
        manifest = Manifest(project="empty")
        result = format_terminal(manifest)
        assert result == "No modules in manifest."

    def test_contains_project_name(self):
        manifest = _make_manifest(("core.engine", _make_entry()))
        result = format_terminal(manifest)
        assert "test_project" in result

    def test_contains_module_name(self):
        manifest = _make_manifest(("core.engine", _make_entry(tier=1)))
        result = format_terminal(manifest)
        assert "core.engine" in result

    def test_contains_tier_label(self):
        manifest = _make_manifest(("core.engine", _make_entry(tier=1)))
        result = format_terminal(manifest)
        # Tier 1 label
        assert "Augmented Pure Python" in result

    def test_tier2_label(self):
        manifest = _make_manifest(("core.engine", _make_entry(tier=2)))
        result = format_terminal(manifest)
        assert "Annotate-Guided Migration" in result

    def test_tier3_label(self):
        manifest = _make_manifest(("core.engine", _make_entry(tier=3)))
        result = format_terminal(manifest)
        assert "Rust Migration" in result

    def test_contains_status(self):
        manifest = _make_manifest(("core.engine", _make_entry(status="analyzed")))
        result = format_terminal(manifest)
        assert "analyzed" in result

    def test_na_score_when_no_annotate_score(self):
        manifest = _make_manifest(("core.engine", _make_entry()))
        result = format_terminal(manifest)
        assert "N/A" in result

    def test_score_present_when_annotate_score_set(self):
        score = AnnotateScore(total_lines=100, yellow_lines=30, score=0.300, hotspots=())
        manifest = _make_manifest(("core.engine", _make_entry(annotate_score=score)))
        result = format_terminal(manifest)
        assert "0.300" in result

    def test_multiple_modules(self):
        manifest = _make_manifest(
            ("core.engine", _make_entry(tier=1)),
            ("core.parser", _make_entry(tier=2)),
        )
        result = format_terminal(manifest)
        assert "core.engine" in result
        assert "core.parser" in result

    def test_returns_string(self):
        manifest = _make_manifest(("mod", _make_entry()))
        assert isinstance(format_terminal(manifest), str)


# ---------------------------------------------------------------------------
# format_json
# ---------------------------------------------------------------------------


class TestFormatJson:
    def test_returns_valid_json(self):
        manifest = _make_manifest(("core.engine", _make_entry()))
        result = format_json(manifest)
        parsed = json.loads(result)  # must not raise
        assert parsed is not None

    def test_has_modules_key(self):
        manifest = _make_manifest(("core.engine", _make_entry()))
        result = format_json(manifest)
        parsed = json.loads(result)
        assert "modules" in parsed

    def test_modules_contains_entry(self):
        manifest = _make_manifest(("core.engine", _make_entry(tier=2)))
        result = format_json(manifest)
        parsed = json.loads(result)
        assert "core.engine" in parsed["modules"]
        assert parsed["modules"]["core.engine"]["tier"] == 2

    def test_project_field_present(self):
        manifest = _make_manifest(("mod", _make_entry()))
        parsed = json.loads(format_json(manifest))
        assert parsed["project"] == "test_project"

    def test_empty_manifest_produces_valid_json(self):
        manifest = Manifest(project="empty")
        result = format_json(manifest)
        parsed = json.loads(result)
        assert parsed["modules"] == {}

    def test_schema_version_present(self):
        manifest = Manifest(project="proj")
        parsed = json.loads(format_json(manifest))
        assert "schema_version" in parsed


# ---------------------------------------------------------------------------
# format_github_issue
# ---------------------------------------------------------------------------


class TestFormatGithubIssue:
    def test_returns_tuple(self):
        entry = _make_entry(tier=1)
        result = format_github_issue("core.engine", entry)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_title_contains_module_key(self):
        entry = _make_entry(tier=1)
        title, _ = format_github_issue("core.engine", entry)
        assert "core.engine" in title

    def test_title_contains_migration_prefix(self):
        entry = _make_entry(tier=1)
        title, _ = format_github_issue("core.engine", entry)
        assert title.startswith("[Migration]")

    def test_title_contains_tier_number(self):
        entry = _make_entry(tier=2)
        title, _ = format_github_issue("core.engine", entry)
        assert "Tier 2" in title

    def test_title_contains_migration_path(self):
        entry = _make_entry(tier=1)
        title, _ = format_github_issue("core.engine", entry)
        assert "Augmented Pure Python" in title

    def test_body_contains_module_key(self):
        entry = _make_entry(tier=1)
        _, body = format_github_issue("core.engine", entry)
        assert "core.engine" in body

    def test_body_contains_source_path(self):
        entry = _make_entry(tier=1)
        _, body = format_github_issue("core.engine", entry)
        assert entry.source_path in body

    def test_body_contains_status(self):
        entry = _make_entry(tier=1, status="analyzed")
        _, body = format_github_issue("core.engine", entry)
        assert "analyzed" in body

    def test_body_contains_annotate_score_when_present(self):
        score = AnnotateScore(total_lines=100, yellow_lines=40, score=0.4, hotspots=())
        entry = _make_entry(tier=1, annotate_score=score)
        _, body = format_github_issue("core.engine", entry)
        assert "0.400" in body or "Annotate Score" in body

    def test_rust_candidate_body_contains_rust_mention(self):
        entry = _make_entry(
            tier=3,
            rust_candidate=True,
            rust_rationale="Heavy C++ interop",
        )
        _, body = format_github_issue("core.engine", entry)
        assert "Rust" in body

    def test_rust_rationale_in_body(self):
        entry = _make_entry(
            tier=3,
            rust_candidate=True,
            rust_rationale="Heavy C++ interop",
        )
        _, body = format_github_issue("core.engine", entry)
        assert "Heavy C++ interop" in body

    def test_no_rust_section_when_not_candidate(self):
        entry = _make_entry(tier=1, rust_candidate=False)
        _, body = format_github_issue("core.engine", entry)
        # Should not contain the Rust section header
        assert "Rust Migration" not in body

    def test_hotspots_included_when_annotate_score_has_hotspots(self):
        h = Hotspot(function="fast_add", line=10, interaction_type="numpy", score=0.9)
        score = AnnotateScore(total_lines=100, yellow_lines=50, score=0.5, hotspots=(h,))
        entry = _make_entry(tier=2, annotate_score=score)
        _, body = format_github_issue("core.engine", entry)
        assert "fast_add" in body

    def test_tier2_title_has_annotate_label(self):
        entry = _make_entry(tier=2)
        title, _ = format_github_issue("mod.sub", entry)
        assert "Annotate-Guided Migration" in title


# ---------------------------------------------------------------------------
# create_github_issue
# ---------------------------------------------------------------------------


class TestCreateGithubIssue:
    def test_skips_when_github_issue_already_set(self, tmp_path: Path):
        """Idempotency: do not create issue if already set."""
        entry = _make_entry(tier=1, github_issue=42)
        manifest = _make_manifest(("core.engine", entry))
        manifest_path = tmp_path / "manifest.json"
        save(manifest, manifest_path)

        result = create_github_issue("owner/repo", "core.engine", entry, manifest, manifest_path)
        assert result is None

    def test_dry_run_returns_none(self, tmp_path: Path):
        entry = _make_entry(tier=1)
        manifest = _make_manifest(("core.engine", entry))
        manifest_path = tmp_path / "manifest.json"
        save(manifest, manifest_path)

        result = create_github_issue(
            "owner/repo",
            "core.engine",
            entry,
            manifest,
            manifest_path,
            dry_run=True,
        )
        assert result is None

    def test_dry_run_does_not_modify_manifest(self, tmp_path: Path):
        entry = _make_entry(tier=1)
        manifest = _make_manifest(("core.engine", entry))
        manifest_path = tmp_path / "manifest.json"
        save(manifest, manifest_path)

        create_github_issue(
            "owner/repo",
            "core.engine",
            entry,
            manifest,
            manifest_path,
            dry_run=True,
        )

        from cython_framework.manifest import load

        reloaded = load(manifest_path)
        assert reloaded.modules["core.engine"].github_issue is None

    def test_raises_when_gh_not_found(self, tmp_path: Path, monkeypatch):
        """RuntimeError when gh CLI is not available."""
        import shutil

        monkeypatch.setattr(shutil, "which", lambda _: None)
        entry = _make_entry(tier=1)
        manifest = _make_manifest(("core.engine", entry))
        manifest_path = tmp_path / "manifest.json"
        save(manifest, manifest_path)

        with pytest.raises(RuntimeError, match="gh"):
            create_github_issue("owner/repo", "core.engine", entry, manifest, manifest_path)
