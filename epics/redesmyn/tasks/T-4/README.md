---
rn:
  node:
    branch: rn/redesmyn/task-4-remove-task-authority
  parent: T-3
---

# T-4 Remove task authority

## Plan

- Remove the `TaskAuthority` construct from the domain/DB/API.
- Replace it with a clearer model (likely `source` + integration refs) and update sync + dashboard accordingly.
