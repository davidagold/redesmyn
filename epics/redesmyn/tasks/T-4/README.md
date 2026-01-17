---
id: T-4
stacked_on: T-3
node:
  branch: rn/redesmyn/task-4-remove-task-authority
---

# T-4 Remove task authority

## Brief (local)

- Remove the `TaskAuthority` construct from the domain/DB/API.
- Replace it with a clearer model (likely `source` + integration refs) and update sync + dashboard accordingly.
