#!/usr/bin/env python3
"""
push.py — Push ChiralCheck to GitHub.
Run from inside the chiralcheck folder.

Usage:
  python push.py
  python push.py --message "your commit message"
  python push.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_URL    = "https://github.com/Ashioya-ui/ChiralCheck.git"
BRANCH      = "main"
DEFAULT_MSG = "feat: add Epistemic Immune System (immune.py) + Dispatch agent gate"

def run(cmd, dry=False):
    print(f"\n  $ {cmd}")
    if dry:
        print(f"    [dry-run]")
        return 0, "", ""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout.strip(): print(f"    {r.stdout.strip()}")
    if r.stderr.strip(): print(f"    {r.stderr.strip()}")
    return r.returncode, r.stdout, r.stderr

def check_prereqs():
    rc, _, _ = run("git --version", dry=False)
    if rc != 0:
        print("\n  ERROR: git not found. Install from git-scm.com\n")
        sys.exit(1)
    if not Path("immune.py").exists():
        print(f"\n  ERROR: Run from inside the chiralcheck folder (where immune.py lives).")
        print(f"  Current folder: {Path.cwd()}\n")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", "-m", default=DEFAULT_MSG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dry = args.dry_run
    msg = args.message

    print("\n  ChiralCheck push")
    print(f"  Repo:    {REPO_URL}")
    print(f"  Branch:  {BRANCH}")
    print(f"  Message: {msg}")
    if dry: print("  Mode:    DRY RUN")

    check_prereqs()

    # 1. Init git repo if this folder has never been initialised
    rc, _, _ = run("git rev-parse --is-inside-work-tree", dry=False)
    if rc != 0:
        print("\n  No .git folder found — initialising repo...")
        rc, _, _ = run("git init", dry)
        if rc != 0 and not dry:
            print("  ERROR: git init failed."); sys.exit(1)
        run(f"git branch -M {BRANCH}", dry)

    # 2. Configure git user if not set (required for first commit on some systems)
    rc, out, _ = run("git config user.email", dry=False)
    if not out.strip():
        run('git config user.email "shoal@ashioya.dev"', dry)
        run('git config user.name "William Ashioya"', dry)

    # 3. Add remote if missing
    rc, out, _ = run("git remote get-url origin", dry=False)
    if rc != 0:
        print("\n  Adding remote origin...")
        rc, _, _ = run(f"git remote add origin {REPO_URL}", dry)
        if rc != 0 and not dry:
            print("  ERROR: Could not add remote."); sys.exit(1)
    else:
        current = out.strip()
        if "ChiralCheck" not in current:
            print(f"\n  WARNING: remote is '{current}'")
            if input("  Continue? [y/N]: ").lower() != "y":
                sys.exit(0)

    # 4. Ensure on main branch
    rc, out, _ = run("git branch --show-current", dry=False)
    if out.strip() not in (BRANCH, ""):
        run(f"git checkout -B {BRANCH}", dry)

    # 5. Stage all files
    print("\n  Staging files...")
    run("git add .", dry)

    # 6. Show what's staged
    print("\n  Staged:")
    subprocess.run("git status --short", shell=True)

    # 7. Commit
    rc, _, err = run(f'git commit -m "{msg}"', dry)
    if rc != 0 and not dry:
        if "nothing to commit" in err.lower():
            print("\n  Nothing new to commit — already up to date.")
        else:
            print("\n  ERROR: Commit failed."); sys.exit(1)

    # 8. Push
    print("\n  Pushing to GitHub...")
    rc, _, err = run(f"git push -u origin {BRANCH}", dry)
    if rc != 0 and not dry:
        if "Authentication" in err or "credential" in err.lower() or "403" in err:
            print("\n  Authentication failed. Fix:")
            print("  Option A — GitHub CLI:")
            print("    gh auth login")
            print("  Option B — Personal Access Token:")
            print("    1. Go to: github.com/settings/tokens")
            print("    2. Fine-grained tokens → New token")
            print("    3. Repo: Ashioya-ui/ChiralCheck  →  Contents: Read+Write")
            print("    4. When git asks for password, paste the token")
        elif "rejected" in err.lower() or "fetch first" in err.lower():
            print("\n  Remote has commits not in local. Pulling first...")
            run(f"git pull --rebase origin {BRANCH}", dry)
            rc, _, _ = run(f"git push -u origin {BRANCH}", dry)
            if rc != 0 and not dry:
                print("  ERROR: Push still failing after rebase.")
                sys.exit(1)
        else:
            print(f"\n  Push failed. See error above.")
            sys.exit(1)

    if not dry:
        print(f"\n  Live at: {REPO_URL.replace('.git', '')}\n")
    else:
        print(f"\n  Dry run complete. Run without --dry-run to execute.\n")

if __name__ == "__main__":
    main()
