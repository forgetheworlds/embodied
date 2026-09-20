"""Mount the local design branch and install its small agent entry file."""

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    design = root / "design"
    if not design.exists():
        if subprocess.run(["git", "-C", str(root), "show-ref", "--verify", "--quiet", "refs/heads/design"]).returncode:
            raise RuntimeError("Local design branch is missing. Obtain it explicitly, or use the public checkout without agent context.")
        subprocess.run(["git", "-C", str(root), "worktree", "add", str(design), "design"], check=True)
    expected = Path(git(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    actual = Path(git(design, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    if expected.resolve() != actual.resolve() or git(design, "branch", "--show-current") != "design":
        raise RuntimeError("design/ must be this repository's design-branch worktree; existing content was not replaced.")
    content = (design / "docs/templates/AGENTS.md").read_text()
    target = root / "AGENTS.md"
    if target.is_symlink():
        raise RuntimeError("AGENTS.md is a symlink; inspect its ownership before replacing it.")
    if target.exists() and target.read_text() != content:
        if not args.replace:
            raise RuntimeError("Different AGENTS.md exists. Review it first; --replace saves a backup.")
        backup = root / "work/agent-context-backups" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup.mkdir(parents=True)
        shutil.copy2(target, backup / "AGENTS.md")
        print("Previous instructions:", backup / "AGENTS.md")
    target.write_text(content)
    print("Agent entry installed:", target)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Setup failed:", error, file=sys.stderr)
        sys.exit(1)
