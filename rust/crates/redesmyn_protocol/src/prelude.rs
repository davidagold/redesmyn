pub const BUILT_IN_PRELUDE_TEMPLATE: &str = "Redesmyn agent prelude\n\
\n\
Assignment\n\
\n\
- You are assigned task {task_id}: {task_title}.\n\
- Read the task doc at {task_doc} and implement its requirements.\n\
\n\
Objective\n\
\n\
- Complete the task end-to-end: implement, validate, and leave the branch in a clean state.\n\
\n\
Context\n\
\n\
- Epic: {epic_slug} (read {epic_readme})\n\
- Branch: {branch}\n\
- Worktree: {worktree}\n\
\n\
Process\n\
\n\
- Read AGENTS.md at repo root and follow it.\n\
- Read the epic README and the task README before coding.\n\
- Skim related tasks (parent, children, blockers) to understand context and avoid conflicts.\n\
- Keep changes small, well-typed, and easy to review; avoid unrelated changes.\n\
- If requirements or context are unclear, ask before making big assumptions.\n\
\n\
Requirements\n\
\n\
- Run `just check` before you finish.\n";

#[derive(Debug, Clone, Copy)]
pub struct PreludeTemplateContext<'a> {
    pub task_id: &'a str,
    pub task_title: &'a str,
    pub task_doc: &'a str,
    pub epic_slug: &'a str,
    pub epic_readme: &'a str,
    pub branch: &'a str,
    pub worktree: &'a str,
}

#[must_use]
pub fn render_prelude_template(template: &str, context: PreludeTemplateContext<'_>) -> String {
    template
        .replace("{task_id}", context.task_id)
        .replace("{task_title}", context.task_title)
        .replace("{task_doc}", context.task_doc)
        .replace("{epic_slug}", context.epic_slug)
        .replace("{epic_readme}", context.epic_readme)
        .replace("{branch}", context.branch)
        .replace("{worktree}", context.worktree)
}

#[must_use]
pub fn render_built_in_prelude(context: PreludeTemplateContext<'_>) -> String {
    render_prelude_template(BUILT_IN_PRELUDE_TEMPLATE, context)
}

#[must_use]
pub fn default_worktree_relative_path(branch_name: &str) -> String {
    let mut out = String::from(".redesmyn/worktrees");
    let mut has_part = false;

    for part in branch_name.split('/') {
        if part.is_empty() || part == "." || part == ".." {
            continue;
        }
        out.push('/');
        out.push_str(&part.replace(':', "_"));
        has_part = true;
    }

    if !has_part {
        let sanitized = branch_name.trim().replace(':', "_");
        if !sanitized.is_empty() {
            out.push('/');
            out.push_str(&sanitized);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::default_worktree_relative_path;

    #[test]
    fn default_worktree_relative_path_uses_branch_components() {
        assert_eq!(
            default_worktree_relative_path("rn/director-v0/T-1-director-run-semantics"),
            ".redesmyn/worktrees/rn/director-v0/T-1-director-run-semantics"
        );
    }

    #[test]
    fn default_worktree_relative_path_skips_unsafe_components() {
        assert_eq!(
            default_worktree_relative_path("../rn/./a:b"),
            ".redesmyn/worktrees/rn/a_b"
        );
    }
}
