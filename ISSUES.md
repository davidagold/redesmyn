# Omnibus Issues (Temporary)

This file is a scratchpad for capturing UX/behavior issues that should be triaged into proper epics/tasks. Once an issue is picked up, move it to the relevant epic/task README and delete it from here.

## 1) Stop running agents on successful merge (spine + ancestors)

**Request:** When a task successfully merges, stop any running agent attached to that task. This should also apply to any **spine/ancestor** tasks that are merged as part of a single merge run.

**Why:** After a merge, the work is “done” from a git/task perspective; leaving agents running is confusing and can keep making edits on branches we just fast-forwarded / restacked.

**Notes / constraints:**
- “Successful merge” here means the branch was fast-forwarded to its upstream as intended (and the merge run step completed), not merely that the merge run started.
- For merge runs that merge multiple spine tasks, stop agents for each task as its merge step completes (or at least by the time the merge run is `succeeded`).
- Must be safe in “dangerous” mode: still enforce git-clean checks before merge execution; this change is about post-merge cleanup.

**Open questions to resolve when implementing:**
- Should we stop only `running` agents, or also `blocked` ones?
- If a merge run is `merge_then_restack` and later restack fails, should we still stop spine-task agents that already merged?

## 2) Replace allow-running modal confirmation with inline confirmation UI

**Request:** The current confirmation modal (“This merge affects running tasks/agents. Proceed anyway?”) is too interruptive. Replace it with an inline confirmation affordance (e.g., secondary button group / menu) near the user’s action point. Include spinners/loading states where needed.

**Why:** Modals are disruptive and can contribute to “graph disappeared” / layout thrash issues. Inline confirmation keeps context and avoids blocking UI interaction.

**Candidate UX direction:**
- In the task action dropdown (Merge / Merge and Restack / Merge then Restack), if the server responds with a 409 “allowRunning required”, replace the clicked menu item with inline “Proceed / Cancel” actions (or show a small anchored confirmation panel).
- For “Resume merge” (both under-card and Details panel), replace the modal with an inline “Proceed anyway” confirmation (same styling language as other under-card affordances).
- Keep the confirmation *scoped* to the specific action, and auto-dismiss it on success/cancel.

**Implementation notes:**
- When this is implemented, do it in a **separate commit** from other merge-related work so it can be reviewed/iterated independently.

