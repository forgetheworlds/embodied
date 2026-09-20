"""Validate public layout and optional local agent context without inference."""

import argparse
from pathlib import Path
import subprocess
import sys

from setup_agent import git


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-context", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    errors = []
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0")
    forbidden = {"design", "archive", "docs", "work", ".atomic", ".Codex", ".codex"}
    for name in filter(None, tracked):
        p = Path(name)
        if p.parts[0] in forbidden or p.name in {"AGENTS.md", "CLAUDE.md"}:
            errors.append("Private context tracked on code branch: " + name)
        if p.suffix.lower() == ".md" and p.name != "README.md":
            errors.append("Public prose belongs in README.md: " + name)
    if git(root, "branch", "--show-current") == "design":
        errors.append("Run from the code checkout, not the design branch.")
    if args.agent_context:
        design = root / "design"
        if not (design / ".git").is_file():
            errors.append("design/ is not a linked worktree. Run setup_agent.py.")
        else:
            common = git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
            other = git(design, "rev-parse", "--path-format=absolute", "--git-common-dir")
            if Path(common).resolve() != Path(other).resolve() or git(design, "branch", "--show-current") != "design":
                errors.append("Design worktree uses the wrong repository or branch.")
        words = 0
        for name in ["START-HERE.md", "GOAL.md", "CURRENT-STATE.md", "DECISIONS.md"]:
            p = design / "docs" / name
            if not p.is_file():
                errors.append("Missing startup document: " + str(p))
            else:
                words += len(p.read_text().split())
        if words > 1800:
            errors.append("Startup packet exceeds 1800 words; move task-specific detail behind links.")
        entry, template = root / "AGENTS.md", design / "docs/templates/AGENTS.md"
        if not entry.is_file() or not template.is_file() or entry.read_text() != template.read_text():
            errors.append("Local AGENTS.md does not match its reviewed template.")
        print("Startup packet:", words, "words (not a tokenizer estimate)")
    for error in errors:
        print("FAIL:", error, file=sys.stderr)
    if errors:
        return 1
    print("PASS: repository foundation checks. No drone capability was tested.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Check failed:", error, file=sys.stderr)
        sys.exit(1)
