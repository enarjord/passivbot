from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = REPO_ROOT / "docs" / "ai"
MANDATORY_FILES = (
    REPO_ROOT / "AGENTS.md",
    AI_ROOT / "principles.md",
    AI_ROOT / "README.md",
)
MANDATORY_WORD_CEILING = 1_500
HAND_AUTHORED_WORD_WARNING = 11_500


@dataclass(frozen=True)
class Issue:
    level: str
    path: Path
    message: str


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _ai_files() -> list[Path]:
    routed = [*AI_ROOT.rglob("*.md"), *AI_ROOT.rglob("*.yaml"), *AI_ROOT.rglob("*.yml")]
    return [REPO_ROOT / "AGENTS.md", REPO_ROOT / "CLAUDE.md", *sorted(set(routed))]


def _resolve_reference(source: Path, value: str) -> Path:
    if value.startswith("docs/") or value in {"AGENTS.md", "CLAUDE.md"}:
        return REPO_ROOT / value
    return source.parent / value


def check_ai_docs() -> list[Issue]:
    issues: list[Issue] = []
    files = _ai_files()

    mandatory_words = sum(len(path.read_text(encoding="utf-8").split()) for path in MANDATORY_FILES)
    if mandatory_words > MANDATORY_WORD_CEILING:
        issues.append(
            Issue(
                "warning",
                AI_ROOT / "README.md",
                f"mandatory context is {mandatory_words} words; ceiling is {MANDATORY_WORD_CEILING}",
            )
        )

    hand_authored = [path for path in files if AI_ROOT / "generated" not in path.parents]
    hand_words = sum(len(path.read_text(encoding="utf-8").split()) for path in hand_authored)
    if hand_words > HAND_AUTHORED_WORD_WARNING:
        issues.append(
            Issue(
                "warning",
                AI_ROOT,
                f"hand-authored AI documentation is {hand_words} words; review above {HAND_AUTHORED_WORD_WARNING}",
            )
        )

    for path in files:
        text = path.read_text(encoding="utf-8")
        headings = re.findall(r"^##+\s+(.+?)\s*$", text, re.M)
        for heading, count in Counter(headings).items():
            if count > 1:
                issues.append(Issue("warning", path, f"duplicate heading {heading!r} ({count} times)"))

        if AI_ROOT / "generated" not in path.parents:
            for phrase in ("Current Implementation Scope", "Implemented now:"):
                if phrase in text:
                    issues.append(Issue("warning", path, f"progress-ledger phrase in AI contract: {phrase!r}"))

        references = set(re.findall(r"`([^`\n]+\.(?:md|ya?ml))`", text))
        references.update(
            target.split("#", 1)[0]
            for target in re.findall(
                r"\[[^\]]+\]\(([^)]+\.(?:md|ya?ml)(?:#[^)]+)?)\)", text
            )
        )
        for reference in sorted(references):
            if any(marker in reference for marker in ("<", ">", "*", "{")):
                continue
            target = _resolve_reference(path, reference).resolve()
            if not target.exists():
                issues.append(
                    Issue("error", path, f"missing Markdown reference {reference!r} -> {_display(target)}")
                )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Check AI-facing documentation structure")
    parser.parse_args()
    issues = check_ai_docs()
    for issue in issues:
        print(f"{issue.level}: {_display(issue.path)}: {issue.message}")
    errors = sum(issue.level == "error" for issue in issues)
    warnings = sum(issue.level == "warning" for issue in issues)
    print(f"AI documentation check: {errors} error(s), {warnings} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
