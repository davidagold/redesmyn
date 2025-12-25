from __future__ import annotations

from dataclasses import dataclass

from redesmyn.domain.enums import BlockMode

READ_ONLY_SUBCOMMANDS: set[str] = {
    "blame",
    "cat-file",
    "diff",
    "grep",
    "log",
    "ls-files",
    "rev-parse",
    "show",
    "status",
}

LAX_ALLOWED_MUTATING: set[str] = {
    "add",
    "apply",
    "commit",
    "mv",
    "restore",
    "revert",
    "rm",
}

LAX_BLOCKED_MUTATING: set[str] = {
    "branch",
    "cherry-pick",
    "merge",
    "push",
    "rebase",
    "reset",
    "tag",
    "worktree",
    # Open questions in the control doc; default to conservative.
    "checkout",
    "pull",
    "stash",
    "switch",
}

STRICT_ALLOWED: set[str] = READ_ONLY_SUBCOMMANDS | {"fetch"}


@dataclass(frozen=True, slots=True)
class GitBlockDecision:
    allowed: bool
    reason: str | None = None


def _detect_subcommand(args: list[str]) -> str | None:
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in {"-C", "--git-dir", "--work-tree", "-c"}:
            i += 2
            continue
        if arg.startswith("-"):
            i += 1
            continue
        return arg
    return None


def _has_flag(args: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(f"{flag}=") for a in args)


def does_block_git(args: list[str], *, mode: str | BlockMode) -> GitBlockDecision:
    """
    Decide whether a `git` invocation should be blocked under a block.

    Modes:
    - lax: allow limited local progress; block push + rewrites/cross-branch ops.
    - strict: block all mutating ops (fetch is allowed).
    """
    subcommand = _detect_subcommand(args)
    if subcommand is None:
        return GitBlockDecision(allowed=True)

    mode_value = mode.value if isinstance(mode, BlockMode) else mode

    if mode_value == "strict":
        if subcommand in STRICT_ALLOWED:
            return GitBlockDecision(allowed=True)
        return GitBlockDecision(
            allowed=False, reason=f"`git {subcommand}` blocked by strict block"
        )

    if mode_value == "lax":
        if subcommand in READ_ONLY_SUBCOMMANDS or subcommand == "fetch":
            return GitBlockDecision(allowed=True)

        if subcommand == "commit" and _has_flag(args, "--amend"):
            return GitBlockDecision(
                allowed=False, reason="`git commit --amend` blocked by lax block"
            )

        if subcommand in LAX_ALLOWED_MUTATING:
            return GitBlockDecision(allowed=True)

        if subcommand in LAX_BLOCKED_MUTATING:
            return GitBlockDecision(
                allowed=False, reason=f"`git {subcommand}` blocked by lax block"
            )

        return GitBlockDecision(
            allowed=False, reason=f"`git {subcommand}` blocked by lax block"
        )

    return GitBlockDecision(allowed=False, reason=f"Unknown block mode: {mode_value}")
