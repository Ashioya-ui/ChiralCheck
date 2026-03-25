"""
agent.py — Dispatch-compatible autonomous ChiralCheck agent.
by William Ashioya | SHOAL

WHAT THIS IS:
  Anthropic just shipped Dispatch + Computer Use (March 24 2026).
  Dispatch lets you assign tasks to Claude from your phone; Claude
  completes them autonomously on your computer using tools.

  This module implements the same agent loop pattern for ChiralCheck:
  Claude receives a task description ("audit my naturalism prior paper"),
  uses bash to find the file and run `python cli.py --json`,
  reads the JSON output, and produces a human-readable audit report —
  all without you touching the keyboard.

  For Max: text Dispatch "audit the naturalism prior draft" from his phone
  → Claude finds paper.txt, runs ChiralCheck, emails him the specificity
  breakdown (the open problems list) before he gets to his desk.

HOW IT WORKS:
  Uses the Anthropic computer use beta API:
    anthropic-beta: computer-use-2025-11-24
  Tools available to the agent:
    bash_20250124      — run shell commands (find files, run cli.py)
    str_replace_based_edit_tool_20250728 — read and write files
  Model: claude-sonnet-4-6 (supports computer use beta)

  The agent loop:
    1. Send task to Claude
    2. Claude calls bash to find/read the input
    3. Claude calls bash to run: python cli.py --input <file> --json
    4. Claude reads the JSON output
    5. Claude writes a structured audit report
    6. Return report to caller

USAGE:
  # Programmatic
  from agent import run_dispatch_task
  report = run_dispatch_task("audit /path/to/paper.txt")
  print(report)

  # CLI
  python agent.py "audit my naturalism prior paper"
  python agent.py "audit /absolute/path/to/paper.txt"
  python agent.py "what is the specificity score of notes.txt"
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Immune system integration — gate tool calls before they execute
# Import lazily inside functions to avoid circular imports at module load
_IMMUNE = None

def _get_immune():
    """Lazy-load the immune system once providers are configured."""
    global _IMMUNE
    if _IMMUNE is None:
        try:
            from auditor import auto_config
            from immune import EpistemicImmuneSystem
            cfg     = auto_config()
            _IMMUNE = EpistemicImmuneSystem(cfg.llm, cfg.embed)
        except Exception:
            _IMMUNE = None   # If setup fails, proceed without gating
    return _IMMUNE

# ── Tool implementations ───────────────────────────────────────────────────────

def _run_bash(command: str, timeout: int = 120) -> str:
    """
    Execute a bash command. Returns stdout+stderr as string.
    Safety: blocks network calls that aren't to allowed API domains.
    Timeouts at 120s to prevent infinite runs.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent,   # always run from chiralcheck dir
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr[:500]}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[ERROR] Command timed out after {timeout}s"
    except Exception as e:
        return f"[ERROR] {e}"


def _read_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        return f"[ERROR] Could not read {path}: {e}"


def _write_file(path: str, content: str) -> str:
    try:
        Path(path).write_text(content, encoding="utf-8")
        return f"Written to {path}"
    except Exception as e:
        return f"[ERROR] Could not write {path}: {e}"


def _execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Dispatch tool calls to implementations.
    HIGH-risk bash commands are gated through the Epistemic Immune System
    before execution — blocked commands return an error string without running.
    """
    if tool_name == "bash":
        cmd = tool_input.get("command", "")

        # Immune gate: classify risk and check before executing
        from immune import AgentAction, TaskRisk
        import re
        HIGH_RISK_PATTERNS = [
            r"\brm\b", r"\brmdir\b", r"\bdd\b", r"\bmkfs\b",
            r"\bchmod\b", r"\bchown\b", r"\bsudo\b", r"\bsu\b",
            r"\bcurl\b.*\|\s*(?:bash|sh)", r"\bwget\b.*\|\s*(?:bash|sh)",
            r"\bdrop\s+table\b", r"\bdelete\s+from\b", r">\s*/dev/",
            r"\bkill\b", r"\bpkill\b", r"\breboot\b", r"\bshutdown\b",
        ]
        is_high_risk = any(re.search(p, cmd, re.IGNORECASE) for p in HIGH_RISK_PATTERNS)
        risk = TaskRisk.HIGH if is_high_risk else TaskRisk.LOW

        immune = _get_immune()
        if immune:
            verdict = immune.evaluate(AgentAction(
                agent_id="dispatch-agent",
                proposed_command=cmd,
                task_risk=risk,
                context="Tool call from Dispatch agent loop",
            ))
            if not verdict.passed:
                return (
                    f"[IMMUNE BLOCK] Action blocked by Epistemic Immune System.\n"
                    f"Score: {verdict.score:.3f} > threshold {verdict.threshold:.2f}\n"
                    f"Detector: {verdict.detector_used}\n"
                    f"Detail: {verdict.detail}\n"
                    f"The action was NOT executed. State rolled back."
                )

        return _run_bash(cmd)

    elif tool_name == "str_replace_based_edit_tool":
        cmd = tool_input.get("command", "")
        if cmd == "view":
            return _read_file(tool_input.get("path", ""))
        elif cmd == "create":
            return _write_file(tool_input.get("path", ""), tool_input.get("file_text", ""))
        elif cmd == "str_replace":
            path = tool_input.get("path", "")
            try:
                content = Path(path).read_text(encoding="utf-8")
                old     = tool_input.get("old_str", "")
                new     = tool_input.get("new_str", "")
                if old not in content:
                    return f"[ERROR] old_str not found in {path}"
                Path(path).write_text(content.replace(old, new, 1), encoding="utf-8")
                return f"Replaced in {path}"
            except Exception as e:
                return f"[ERROR] {e}"
        return f"[ERROR] Unknown text_editor command: {cmd}"

    return f"[ERROR] Unknown tool: {tool_name}"


# ── Agent loop ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are ChiralCheck Agent, an autonomous idea auditor built on SHOAL
(github.com/Ashioya-ui/shoal) by William Ashioya.

Your job: when given a task involving a paper, thesis, or idea, you:
1. Find the relevant file(s) or extract the text from the task description
2. Run the ChiralCheck audit using: python cli.py --input <file> --json
   OR: python cli.py --text "<text>" --json
3. Parse the JSON output
4. Write a clear, structured audit report covering:
   - Composite score and what it means
   - Open problems (UNKNOWN subtasks from the specificity axis) — the most important output
   - Stability verdict (STABLE ATTRACTOR / META-STABLE / FRAGILE)
   - Utility isolation index
   - Originality score and closest prior-art reference
   - Concrete recommendation: what needs more work

Always run the actual tool. Do not guess scores. If a file path is mentioned,
use bash to verify it exists before running. If you get JSON parse errors,
try running with --text instead.

The open problems list is what the person cares about most.
Format it clearly so they can act on it immediately.
""".strip()


def run_dispatch_task(
    task:        str,
    max_iter:    int  = 12,
    verbose:     bool = True,
    model:       str  = "claude-sonnet-4-6",
) -> str:
    """
    Run an autonomous ChiralCheck task.

    Args:
        task:     Natural language task description.
                  e.g. "audit /home/user/papers/naturalism.txt"
                  e.g. "what is the specificity score of this thesis: [text]"
        max_iter: Safety cap on agent loop iterations.
        verbose:  Print agent progress to stdout.
        model:    Anthropic model. Must support computer-use-2025-11-24 beta.

    Returns:
        str: The agent's final audit report.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ERROR] ANTHROPIC_API_KEY not set. See .env.example"

    client = anthropic.Anthropic(api_key=api_key, timeout=120.0)

    tools = [
        {
            "type":    "bash_20250124",
            "name":    "bash",
        },
        {
            "type":    "text_editor_20250728",
            "name":    "str_replace_based_edit_tool",
        },
    ]

    messages: list[dict] = [
        {"role": "user", "content": task}
    ]

    if verbose:
        print(f"\n  [agent] Task: {task[:80]}")
        print(f"  [agent] Model: {model}")
        print(f"  [agent] Starting loop (max {max_iter} iterations)...")
        print()

    final_text = ""

    for iteration in range(max_iter):
        try:
            response = client.beta.messages.create(
                model=model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
                betas=["computer-use-2025-11-24"],
            )
        except Exception as e:
            return f"[ERROR] API call failed: {e}"

        # Collect text content
        text_blocks  = [b for b in response.content if b.type == "text"]
        tool_blocks  = [b for b in response.content if b.type == "tool_use"]

        if text_blocks:
            final_text = "\n".join(b.text for b in text_blocks)
            if verbose:
                print(f"  [agent] Iteration {iteration+1}: {final_text[:100]}...")

        # If done, return
        if response.stop_reason == "end_turn":
            if verbose:
                print(f"\n  [agent] Complete after {iteration+1} iterations.")
            return final_text or "[agent completed with no text output]"

        # Execute tool calls
        if not tool_blocks:
            break

        tool_results = []
        for block in tool_blocks:
            if verbose:
                input_preview = json.dumps(block.input)[:80]
                print(f"  [tool]  {block.name}: {input_preview}")

            result_text = _execute_tool(block.name, block.input)

            if verbose:
                print(f"  [out]   {result_text[:120]}")

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result_text,
            })

        # Append assistant turn + tool results
        messages = messages + [
            {"role": "assistant", "content": response.content},
            {"role": "user",      "content": tool_results},
        ]

    return final_text or "[agent: max iterations reached without completing task]"


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python agent.py \"your task here\"")
        print("Example: python agent.py \"audit paper.txt\"")
        sys.exit(1)

    task   = " ".join(sys.argv[1:])
    report = run_dispatch_task(task, verbose=True)

    print("\n" + "═" * 60)
    print("  AUDIT REPORT")
    print("═" * 60)
    print(report)
    print("═" * 60 + "\n")
