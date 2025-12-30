# T-1 Conversation continuity + resume tokens (Codex-first)

## Context

In v0 dogfooding, we frequently **restart** agent runners (e.g. when changing sandbox rules, runner wrapper behavior, or tmux lifecycle). Today, a restart effectively loses the in-harness conversation context unless the user manually resumes it.

We intentionally **do not** want to immediately build a full “chat continuity” system that parses/stores a harness continuation token and automatically rehydrates context (this can be harness-specific, privacy-sensitive, and operationally complex).

However, we still need a clear, reliable path to:

- Resume work without “starting from zero” after a restart.
- Make it obvious to the user what continuity guarantees exist (and what don’t).

## Problem statement

When a runner session is restarted, the harness starts with a fresh conversation. Users need a lightweight way to continue the same work without having to manually reconstruct context or paste long prior transcripts.

## Proposed directions (non-binding)

### Option A: Harness-native resume (preferred if available)

If the harness provides a native resume mechanism (e.g., `codex resume`), expose it as a first-class action in the UI/CLI and persist any minimal identifiers required to find the latest session.

### Option B: Minimal continuity via “last rendered prelude”

Persist and surface the rendered prelude used for a run (including task + environment guidance). On restart, allow users to re-send it (or a user-edited variant) as a quick “bootstrap” even without full chat restoration.

### Option C: Optional continuation token persistence (later)

If we later decide to persist a continuation token (harness-specific), it should be:

- Explicitly opt-in.
- Stored in a well-scoped place with clear lifecycle.
- Considered sensitive (avoid accidental sharing).

## Acceptance criteria

- The UI and CLI can “resume” a prior harness session in a way that is explicit and user-controlled (Codex-first).
- The system makes continuity boundaries clear: what persists automatically vs what does not.
- Documentation in the epic explains the tradeoffs and the chosen approach.

