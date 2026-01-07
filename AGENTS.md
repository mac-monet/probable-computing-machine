# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with remote
```

## Session Start

When beginning a new session:

1. `bd ready` - find available work
2. `bd show <id>` - read issue details including `## Verify` section
3. `jj log -r @` - understand current repo state
4. Read relevant spec files referenced in the issue/epic
5. `zig build test` - verify baseline passes before making changes

## Creating Issues

When creating issues, include a `## Verify` section with explicit verification steps:

```markdown
## Description
Implement mulGate for CircuitBuilder

## Verify
- [ ] `zig build test` passes for constraint module
- [ ] Test exists: satisfied constraint (3 * 7 = 21)
- [ ] Test exists: unsatisfied constraint returns non-zero
- [ ] Invalid column index returns error.InvalidColumn
```

Verification steps should be:
- **Mechanical** - runnable commands or specific checks
- **Unambiguous** - clear pass/fail criteria
- **Complete** - cover the key requirements

## Completing an Issue

When you finish an issue:

1. **Verify implementation** - Use the Task tool to spawn a subagent that:
   - Re-reads the issue (`bd show <id>`)
   - Checks each item in the `## Verify` section, reporting pass/fail per step
   - Finds the parent epic and reads the spec file from its `external_ref` field
   - Does broader review against spec for unlisted issues
   - Reports structured results:
     ```
     ## Verification Results
     - [x] `zig build test` passes
     - [x] Satisfied constraint test exists
     - [ ] FAIL: Missing test for unsatisfied constraint

     ## Spec Review
     - Section 4.2: Implemented correctly
     - Section 4.3: Missing error handling for X
     ```
   - If any step fails, address gaps before proceeding
2. **Commit your changes** - `jj commit -m "..."`
3. **Run quality gates** (if code changed) - `zig build test`
4. **Close the issue** - `bd close <id>`
5. **File issues for remaining work** - Create issues for anything out of scope

Do NOT push after individual issues - continue working on the next issue in the epic.

## Landing the Plane (Epic Completion)

**When completing an epic**, you MUST complete ALL steps below. An epic is NOT complete until `jj git push` succeeds.

**MANDATORY WORKFLOW:**

1. **Verify all issues in the epic are closed** - `bd epic show <id>`
2. **Run quality gates** - Tests, linters, builds for the full changeset
3. **Clean up** - Abandon empty changes, clean up bookmarks
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   jj git fetch
   bd sync
   jj git push
   jj log -r 'remote_bookmarks()'  # Verify remote is synced
   ```
5. **Close the epic** - `bd close <epic-id>`
6. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- An epic is NOT complete until `jj git push` succeeds
- NEVER stop before pushing when finishing an epic - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

