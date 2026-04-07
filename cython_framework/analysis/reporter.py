"""Migration manifest reporter - terminal, JSON, and GitHub output formats.

CLI usage::

    python -m cython_framework.analysis.reporter --manifest <path> --output terminal|json|github

"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cython_framework.manifest as manifest_mod
from cython_framework.manifest import Manifest, ModuleEntry, load, replace_module, save

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TIER_LABELS: dict[int, str] = {
    1: "Augmented Pure Python",
    2: "Annotate-Guided Migration",
    3: "Rust Migration",
}

_COL_MODULE = 40
_COL_TIER = 6
_COL_SCORE = 8
_COL_STATUS = 20


# ---------------------------------------------------------------------------
# Format functions
# ---------------------------------------------------------------------------


def format_terminal(manifest: Manifest) -> str:
    """Format manifest as a human-readable terminal table.

    Returns:
        A formatted table string, or "No modules in manifest." if empty.
    """
    if not manifest.modules:
        return "No modules in manifest."

    lines: list[str] = []
    lines.append(f"Project: {manifest.project}")
    lines.append("")

    # Header
    header = (
        f"{'Module':<{_COL_MODULE}} "
        f"{'Tier':>{_COL_TIER}} "
        f"{'Score':>{_COL_SCORE}} "
        f"{'Status':<{_COL_STATUS}}"
    )
    separator = "-" * (len(header))
    lines.append(header)
    lines.append(separator)

    for key, entry in sorted(manifest.modules.items()):
        score_str = (
            f"{entry.annotate_score.score:.3f}" if entry.annotate_score is not None else "   N/A"
        )
        tier_label = _TIER_LABELS.get(entry.tier, str(entry.tier))
        # Truncate module key if too long
        display_key = key if len(key) <= _COL_MODULE else key[: _COL_MODULE - 3] + "..."
        line = (
            f"{display_key:<{_COL_MODULE}} "
            f"{tier_label!s:>{_COL_TIER}} "
            f"{score_str:>{_COL_SCORE}} "
            f"{entry.status:<{_COL_STATUS}}"
        )
        lines.append(line)

    return "\n".join(lines)


def format_json(manifest: Manifest) -> str:
    """Serialize manifest to a JSON string.

    Returns:
        Valid JSON string representation of the manifest.
    """
    data = manifest_mod._manifest_to_dict(manifest)  # noqa: SLF001
    return json.dumps(data, indent=2, sort_keys=True)


def format_github_issue(module_key: str, entry: ModuleEntry) -> tuple[str, str]:
    """Format a GitHub issue title and body for a migration module.

    Args:
        module_key: Dotted module key (e.g. ``core.engine``).
        entry: The :class:`ModuleEntry` for this module.

    Returns:
        A ``(title, body)`` tuple ready for ``gh issue create``.
    """
    migration_path = _TIER_LABELS.get(entry.tier, f"Tier {entry.tier}")
    title = f"[Migration] {module_key} - Tier {entry.tier} {migration_path}"

    body_lines: list[str] = [
        f"## Module: `{module_key}`",
        "",
        f"**Tier:** {entry.tier} – {migration_path}",
        f"**Source Path:** `{entry.source_path}`",
        f"**Status:** `{entry.status}`",
    ]

    if entry.annotate_score is not None:
        a = entry.annotate_score
        body_lines += [
            "",
            "### Annotate Score",
            f"- Score: `{a.score:.3f}`",
            f"- Total lines: {a.total_lines}",
            f"- Yellow lines: {a.yellow_lines}",
        ]
        if a.hotspots:
            body_lines.append("")
            body_lines.append("**Top hotspots:**")
            for h in a.hotspots[:5]:
                body_lines.append(
                    f"  - `{h.function}` (line {h.line}): {h.interaction_type} score={h.score:.2f}"
                )

    if entry.rust_candidate:
        body_lines += [
            "",
            "### Rust Migration",
            "> This module has been flagged as a **Rust migration candidate**.",
        ]
        if entry.rust_rationale:
            body_lines.append(f"> Rationale: {entry.rust_rationale}")

    body_lines += [
        "",
        "---",
        "_This issue was generated automatically by hb-cython-framework._",
    ]

    return title, "\n".join(body_lines)


# ---------------------------------------------------------------------------
# GitHub issue creation
# ---------------------------------------------------------------------------


def create_github_issue(
    repo: str,
    module_key: str,
    entry: ModuleEntry,
    manifest: Manifest,
    manifest_path: str | Path,
    *,
    dry_run: bool = False,
) -> str | None:
    """Create a GitHub issue for a module migration, updating the manifest on success.

    Idempotent: skips creation if ``entry.github_issue`` is already set.

    Args:
        repo: GitHub repository in ``owner/repo`` format.
        module_key: Dotted module key.
        entry: The :class:`ModuleEntry` for this module.
        manifest: The full :class:`Manifest` (used for updates).
        manifest_path: Path to the manifest file (updated with issue URL).
        dry_run: If ``True``, print the issue details but do not create it.

    Returns:
        The created issue URL string, or ``None`` if skipped / dry-run.

    Raises:
        RuntimeError: If the ``gh`` CLI is not found.
        subprocess.CalledProcessError: If the ``gh`` command fails.
    """
    if entry.github_issue is not None:
        return None

    title, body = format_github_issue(module_key, entry)

    if dry_run:
        print("[dry-run] Would create GitHub issue:")
        print(f"  Title: {title}")
        print(f"  Body:\n{body}")
        return None

    if shutil.which("gh") is None:
        raise RuntimeError(
            "GitHub CLI ('gh') not found. "
            "Install it from https://cli.github.com/ and authenticate with 'gh auth login'."
        )

    result = subprocess.run(  # noqa: S603
        ["gh", "issue", "create", "--repo", repo, "--title", title, "--body", body],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    issue_url = result.stdout.strip()

    # Update manifest with issue URL
    updated_entry = dataclasses.replace(entry, github_issue=issue_url)
    updated_manifest = replace_module(manifest, module_key, updated_entry)
    save(updated_manifest, manifest_path)

    return issue_url


# ---------------------------------------------------------------------------
# CLI (__main__)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report on migration manifest in various formats.",
        prog="python -m cython_framework.analysis.reporter",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        metavar="PATH",
        help="Path to the migration manifest JSON file.",
    )
    parser.add_argument(
        "--output",
        choices=["terminal", "json", "github"],
        default="terminal",
        help="Output format (default: terminal).",
    )
    parser.add_argument(
        "--repo",
        metavar="OWNER/REPO",
        help="GitHub repository (required for --output github).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --output github, print issue details without creating.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    manifest = load(manifest_path)

    if args.output == "terminal":
        print(format_terminal(manifest))

    elif args.output == "json":
        print(format_json(manifest))

    elif args.output == "github":
        if not args.repo:
            parser.error("--repo is required when using --output github")
        for module_key, entry in manifest.modules.items():
            create_github_issue(
                args.repo,
                module_key,
                entry,
                manifest,
                manifest_path,
                dry_run=args.dry_run,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
